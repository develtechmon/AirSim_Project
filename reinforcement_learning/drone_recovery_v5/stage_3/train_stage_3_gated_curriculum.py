"""
PERFORMANCE-GATED CURRICULUM TRAINING
======================================
Trains with automatic difficulty advancement based on performance!

Only advances to next level when current level is MASTERED:
- Easy (0.7-0.9):   80% recovery required
- Medium (0.9-1.1): 70% recovery required
- Hard (1.1-1.5):   60% recovery target (PhD complete)

Usage:
    python train_gated_curriculum.py
"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from drone_flip_recovery_env_injector_gated import DroneFlipRecoveryEnv
import argparse
from pathlib import Path
import time


class GatedCurriculumCallback(BaseCallback):
    """Enhanced callback tracking curriculum progression"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.recovery_successes = []
        self.recovery_times = []
        self.intensities = []
        self.curriculum_levels = []
        self.episode_count = 0
        self.start_time = time.time()
        
        # Track when levels were achieved
        self.level_achievements = {}
    
    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_count += 1
                    
                    info = self.locals["infos"][i]
                    ep_reward = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Track recovery stats
                    tumble_initiated = info.get("tumble_initiated", False)
                    tumble_recovered = info.get("tumble_recovered", False)
                    recovery_steps = info.get("recovery_steps", 0)
                    intensity = info.get("disturbance_intensity", 1.0)
                    curriculum_level = info.get("curriculum_level", 0)
                    
                    self.curriculum_levels.append(curriculum_level)
                    
                    # Track level achievements
                    if curriculum_level not in self.level_achievements:
                        self.level_achievements[curriculum_level] = {
                            'reached_at_episode': self.episode_count,
                            'elapsed_time': time.time() - self.start_time
                        }
                    
                    if tumble_initiated:
                        self.recovery_successes.append(1 if tumble_recovered else 0)
                        self.intensities.append(intensity)
                        if tumble_recovered:
                            self.recovery_times.append(recovery_steps)
                    
                    # Print every 10 episodes
                    if self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_lengths = self.episode_lengths[-10:]
                        recent_recoveries = self.recovery_successes[-10:]
                        recent_intensities = self.intensities[-10:]
                        recent_levels = self.curriculum_levels[-10:]
                        
                        recovery_rate = np.mean(recent_recoveries) * 100 if recent_recoveries else 0
                        avg_intensity = np.mean(recent_intensities) if recent_intensities else 0
                        avg_recovery_time = np.mean(self.recovery_times[-10:]) if len(self.recovery_times) > 0 else 0
                        current_level = recent_levels[-1] if recent_levels else 0
                        
                        level_names = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
                        
                        elapsed = time.time() - self.start_time
                        
                        print(f"\n{'='*70}")
                        print(f"📊 EPISODE {self.episode_count} | {elapsed/3600:.1f}h elapsed")
                        print(f"{'='*70}")
                        print(f"   🎓 Curriculum Level: {current_level} - {level_names[current_level]}")
                        print(f"   Last 10 Episodes:")
                        print(f"      Avg Return: {np.mean(recent_rewards):.1f}")
                        print(f"      Avg Length: {np.mean(recent_lengths):.1f} steps")
                        print(f"   Recovery Performance:")
                        print(f"      Recovery Rate: {recovery_rate:.0f}%")
                        print(f"      Avg Intensity: {avg_intensity:.2f}x")
                        if avg_recovery_time > 0:
                            print(f"      Avg Recovery Time: {avg_recovery_time:.0f} steps ({avg_recovery_time*0.05:.1f}s)")
                        
                        # Show progression status
                        if current_level == 0:
                            target = 80
                            print(f"   📈 Progress: {recovery_rate:.0f}% / {target}% (need to advance)")
                        elif current_level == 1:
                            target = 70
                            print(f"   📈 Progress: {recovery_rate:.0f}% / {target}% (need to advance)")
                        else:
                            target = 60
                            print(f"   📈 Progress: {recovery_rate:.0f}% / {target}% (PhD target)")
                        
                        print(f"{'='*70}\n")
        
        return True


def main(args):
    print("\n" + "="*70)
    print("🎓 PERFORMANCE-GATED CURRICULUM TRAINING")
    print("="*70)
    print("Training with automatic difficulty advancement!")
    print("")
    print("Progression Rules:")
    print("  Level 0 (EASY):   0.7-0.9x intensity → Need 80% recovery")
    print("  Level 1 (MEDIUM): 0.9-1.1x intensity → Need 70% recovery")
    print("  Level 2 (HARD):   1.1-1.5x intensity → Target 60% recovery")
    print("")
    print("✅ Only advances when current level is MASTERED!")
    print("="*70 + "\n")
    
    # Create environment
    print(f"[1/3] Creating gated curriculum environment...")
    print(f"   Altitude: 30m")
    print(f"   Max steps: 600")
    print(f"   Strategy: Performance-gated progression")
    
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=30.0,
            max_steps=600,
            wind_strength=args.wind_strength,
            flip_prob=args.flip_prob,
            debug=True
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    
    print("   ✅ Gated curriculum environment created!")
    
    # Load Stage 2 model
    print(f"\n[2/3] Loading Stage 2 trained model...")
    print(f"   Model path: {args.stage2_model}")
    try:
        model = PPO.load(
            args.stage2_model,
            env=env,
            device="cpu"
        )
        print("   ✅ Stage 2 policy loaded!")
        
        if args.lr < 3e-5:
            model.learning_rate = args.lr
            print(f"   📉 Learning rate: {args.lr}")
        
    except Exception as e:
        print(f"   ⚠️  Could not load Stage 2 model: {e}")
        print(f"   ⚠️  Training from scratch")
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/gated_curriculum/",
            device="cpu",
            policy_kwargs=dict(
                net_arch=[256, 256, 128]
            )
        )
    
    # Setup callbacks
    progress_callback = GatedCurriculumCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/gated_checkpoints/",
        name_prefix="gated_curriculum",
        save_vecnormalize=True
    )
    
    callbacks = [progress_callback, checkpoint_callback]
    
    # Create directories
    Path("./models/gated_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("./logs/gated_curriculum/").mkdir(parents=True, exist_ok=True)
    
    print("\n[3/3] Starting gated curriculum training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {model.learning_rate}")
    print()
    print("="*70)
    print("🚀 GATED TRAINING STARTED")
    print("="*70)
    print("Watch for curriculum advancements:")
    print("  ✅ Level 0 → 1: When easy cases reach 80% recovery")
    print("  ✅ Level 1 → 2: When medium cases reach 70% recovery")
    print("  🎯 Level 2:     Target 60% recovery on hard cases")
    print("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "="*70)
        print("✅ GATED TRAINING COMPLETE!")
        print("="*70)
        
        # Save final model
        model.save("./models/gated_curriculum_policy")
        env.save("./models/gated_curriculum_vecnormalize.pkl")
        
        print(f"\n💾 Model saved:")
        print(f"   - ./models/gated_curriculum_policy.zip")
        print(f"   - ./models/gated_curriculum_vecnormalize.pkl")
        
        print(f"\n📊 Training Statistics:")
        print(f"   Total episodes: {progress_callback.episode_count}")
        if progress_callback.episode_rewards:
            print(f"   Avg return: {np.mean(progress_callback.episode_rewards[-50:]):.1f} (last 50)")
            print(f"   Recovery rate: {np.mean(progress_callback.recovery_successes[-50:])*100:.0f}% (last 50)")
            if len(progress_callback.recovery_times) > 0:
                print(f"   Avg recovery time: {np.mean(progress_callback.recovery_times[-20:]):.0f} steps")
        
        print(f"\n🎓 Curriculum Progression:")
        for level, data in sorted(progress_callback.level_achievements.items()):
            level_names = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
            print(f"   Level {level} ({level_names[level]}): Reached at episode {data['reached_at_episode']} ({data['elapsed_time']/3600:.1f}h)")
        
        print("\n✅ Next: Test with test_stage3_policy.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        model.save("./models/gated_curriculum_policy_interrupted")
        env.save("./models/gated_curriculum_vecnormalize_interrupted.pkl")
        print("💾 Model saved at interruption")
    
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stage2-model', type=str, 
                        default='./models/hover_disturbance_policy_interrupted.zip',
                        help='Path to Stage 2 trained policy')
    parser.add_argument('--timesteps', type=int, default=600000,
                        help='Total training timesteps')
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Maximum wind strength (m/s)')
    parser.add_argument('--flip-prob', type=float, default=1.0,
                        help='Disturbance probability')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    main(args)