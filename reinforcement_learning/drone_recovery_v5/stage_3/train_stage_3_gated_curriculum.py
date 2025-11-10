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

    ## 📁 **FILE STRUCTURE:**
```
models/stage3_checkpoints/
                ├── curriculum_levels/              ← NEW!
                │   ├── level_0_EASY_mastered.zip          ← Episode 203
                │   ├── level_0_EASY_mastered_vecnormalize.pkl
                │   ├── level_0_EASY_mastered_metadata.json
                │   ├── level_1_MEDIUM_mastered.zip        ← Episode 348
                │   ├── level_1_MEDIUM_mastered_vecnormalize.pkl
                │   └── level_1_MEDIUM_mastered_metadata.json
                │
                └── gated_curriculum_policy.zip     ← Final model

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
    """Enhanced callback tracking curriculum progression with AUTO-SAVE"""
    
    def __init__(self, save_path="./models/stage3_checkpoints/curriculum_levels/", verbose=0):
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
        
        # Auto-save setup
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
    
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
                    
                    # CHECK FOR LEVEL ADVANCEMENT (AUTO-SAVE TRIGGER)
                    env = self.training_env.envs[0]
                    if hasattr(env, 'env') and hasattr(env.env, 'level_advanced'):
                        if env.env.level_advanced:
                            # LEVEL ADVANCED! AUTO-SAVE MODEL!
                            advancement_info = env.env.advancement_info
                            self._save_advancement_model(advancement_info)
                            # Reset flag
                            env.env.level_advanced = False
                    
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
    
    def _save_advancement_model(self, advancement_info):
        """Save model when curriculum level advances"""
        old_level = advancement_info['old_level']
        new_level = advancement_info['new_level']
        recovery_rate = advancement_info['recovery_rate']
        episode = advancement_info['episode']
        
        level_names = ["EASY", "MEDIUM", "HARD"]
        
        # Save model
        model_filename = f"level_{old_level}_{level_names[old_level]}_mastered"
        model_path = self.save_path / model_filename
        
        print("\n" + "="*70)
        print("💾 AUTO-SAVING MODEL - LEVEL MASTERED!")
        print("="*70)
        print(f"   Level {old_level} ({level_names[old_level]}) completed")
        print(f"   Recovery rate: {recovery_rate*100:.1f}%")
        print(f"   Episode: {episode}")
        print(f"   Saving to: {model_path}")
        
        # Save the model
        self.model.save(str(model_path))
        
        # Save VecNormalize stats
        vecnorm_path = self.save_path / f"{model_filename}_vecnormalize.pkl"
        if hasattr(self.training_env, 'save'):
            self.training_env.save(str(vecnorm_path))
        
        # Save metadata
        metadata = {
            'level': old_level,
            'level_name': level_names[old_level],
            'recovery_rate': float(recovery_rate),
            'episode': episode,
            'timestamp': time.time() - self.start_time
        }
        
        import json
        metadata_path = self.save_path / f"{model_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model saved: {model_path}.zip")
        print(f"   ✅ VecNormalize saved: {vecnorm_path}")
        print(f"   ✅ Metadata saved: {metadata_path}")
        print("="*70 + "\n")


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
    save_path = "./models/stage3_checkpoints/curriculum_levels/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    progress_callback = GatedCurriculumCallback(save_path=save_path)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/stage3_checkpoints/gated_checkpoints/",
        name_prefix="gated_curriculum",
        save_vecnormalize=True
    )
    
    callbacks = [progress_callback, checkpoint_callback]
    
    # Create directories
    Path("./models/stage3_checkpoints/gated_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("./logs/gated_curriculum/").mkdir(parents=True, exist_ok=True)
    
    print("\n[3/3] Starting gated curriculum training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   💾 Auto-save enabled: {save_path}")
    print()
    print("="*70)
    print("🚀 GATED TRAINING STARTED")
    print("="*70)
    print("Watch for curriculum advancements:")
    print("  ✅ Level 0 → 1: When easy cases reach 80% recovery")
    print("     💾 Auto-saves: level_0_EASY_mastered.zip")
    print("  ✅ Level 1 → 2: When medium cases reach 70% recovery")
    print("     💾 Auto-saves: level_1_MEDIUM_mastered.zip")
    print("  🎯 Level 2:     Target 60% recovery on hard cases")
    print("     💾 Final save: gated_curriculum_policy.zip")
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
        model.save("./models/stage3_checkpoints/gated_curriculum_policy")
        env.save("./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl")
        
        print(f"\n💾 Models saved:")
        print(f"\n   📂 Curriculum Level Models (Auto-saved during training):")
        if (Path(save_path) / "level_0_EASY_mastered.zip").exists():
            print(f"      ✅ Level 0 (EASY):   {save_path}level_0_EASY_mastered.zip")
        if (Path(save_path) / "level_1_MEDIUM_mastered.zip").exists():
            print(f"      ✅ Level 1 (MEDIUM): {save_path}level_1_MEDIUM_mastered.zip")
        
        print(f"\n   📂 Final Model:")
        print(f"      ✅ ./models/stage3_checkpoints/gated_curriculum_policy.zip")
        print(f"      ✅ ./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl")
        
        print(f"\n   📂 Regular Checkpoints (every 50k steps):")
        checkpoint_files = list(Path("./models/stage3_checkpoints/gated_checkpoints/").glob("*.zip"))
        if checkpoint_files:
            for i, ckpt in enumerate(sorted(checkpoint_files)[-3:]):  # Show last 3
                print(f"      ✅ {ckpt}")
        
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
        
        print("\n✅ Next Steps:")
        print("   1. Test overall performance:")
        print("      python test_gated_curriculum.py --episodes 60")
        print("\n   2. Test specific level model:")
        print("      python test_gated_curriculum.py --model ./models/stage3_checkpoints/curriculum_levels/level_0_EASY_mastered.zip")
        print("\n   3. Compare level performances:")
        print("      - Easy model on all intensities")
        print("      - Medium model on all intensities")
        print("      - Final model on all intensities")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        model.save("./models/stage3_checkpoints/gated_curriculum_policy_interrupted")
        env.save("./models/stage3_checkpoints/gated_curriculum_vecnormalize_interrupted.pkl")
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