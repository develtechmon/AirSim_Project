"""
STAGE 3: PPO TRAINING WITH FLIP RECOVERY (13 OBSERVATIONS)
===========================================================
Trains the drone to recover from flips using PPO (Reinforcement Learning).

Key Features:
- Loads your Stage 2 disturbance policy as starting point
- Adds flip recovery scenarios
- Fine-tunes with PPO for 3 hours
- Expected result: 75%+ flip recovery

Usage:
    python train_stage3_flip_v2.py

That's it! Just like the original!

To retrain from your current checkpoint:
python train_stage3_flip_v2.py \
  --stage2-model ./models/flip_recovery_policy.zip \
  --flip-prob 1.0 \
  --timesteps 300000

python stage_3_work_on_lower_intensity/train_stage3_flip_v2.py  --stage2-model ./models/flip_recovery_policy.zip  --flip-prob 1.0 --timesteps 250000

To run default
python train_stage3_flip_v2.py 

"""

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
# from drone_flip_recovery_env import DroneFlipRecoveryEnv
from drone_flip_recovery_env_injector import DroneFlipRecoveryEnv

import argparse
from pathlib import Path


class ProgressCallback(BaseCallback):
    """Callback to log training progress with flip recovery stats"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.recovery_successes = []
        self.recovery_times = []
        self.episode_count = 0
    
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
                    
                    # Only track if tumble happened
                    if tumble_initiated:
                        self.recovery_successes.append(1 if tumble_recovered else 0)
                        if tumble_recovered:
                            self.recovery_times.append(recovery_steps)
                    
                    # Print every 10 episodes
                    if self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_lengths = self.episode_lengths[-10:]
                        recent_recoveries = self.recovery_successes[-10:]
                        
                        recovery_rate = np.mean(recent_recoveries) * 100
                        avg_recovery_time = np.mean(self.recovery_times[-10:]) if len(self.recovery_times) > 0 else 0
                        
                        print(f"\n{'='*70}")
                        print(f"📊 EPISODE {self.episode_count}")
                        print(f"{'='*70}")
                        print(f"   Last 10 Episodes:")
                        print(f"      Avg Return: {np.mean(recent_rewards):.1f}")
                        print(f"      Avg Length: {np.mean(recent_lengths):.1f} steps")
                        print(f"      Max Length: {np.max(recent_lengths)} steps")
                        print(f"   Flip Recovery:")
                        print(f"      Recovery Rate: {recovery_rate:.0f}%")
                        if avg_recovery_time > 0:
                            print(f"      Avg Recovery Time: {avg_recovery_time:.0f} steps")
                        
                        wind_mag = info.get("wind_magnitude", 0)
                        print(f"   Current wind: {wind_mag:.1f} m/s")
                        print(f"{'='*70}\n")
        
        return True


def main(args):
    print("\n" + "="*70)
    print("🔄 STAGE 3: FLIP RECOVERY TRAINING")
    print("="*70)
    print("Training drone to recover from any orientation!")
    print(f"Starting from Stage 2 policy (90%+ hover + wind success)")
    print(f"Expected training time: 3 hours")
    print("="*70 + "\n")
    
    # Create environment50
    print(f"[1/3] Creating flip recovery environment...")
    print(f"   Wind strength: 0-{args.wind_strength} m/s")
    print(f"   Flip probability: {args.flip_prob*100:.0f}%")
    
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=10.0,
            max_steps=500,
            wind_strength=args.wind_strength,
            flip_prob=args.flip_prob,
            debug=True  # Changed to True to see tumble messages!
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create NEW VecNormalize for Stage 3
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    
    print("   ✅ Environment created with flip scenarios")
    
    # Load Stage 2 model
    print(f"\n[2/3] Loading Stage 2 trained model...")
    print(f"   Model path: {args.stage2_model}")
    try:
        # Load Stage 2 PPO model directly
        model = PPO.load(
            args.stage2_model,
            env=env,
            device="cpu"
        )
        print("   ✅ Stage 2 policy loaded successfully!")
        print("   💡 Starting from 90%+ hover + wind success!")
        print("   📈 Will learn flip recovery on top of existing skills")
        
        # Adjust learning rate for fine-tuning
        if args.lr < 3e-5:
            model.learning_rate = args.lr
            print(f"   📉 Learning rate adjusted to {args.lr} for fine-tuning")
        
    except Exception as e:
        print(f"   ⚠️  Could not load Stage 2 model: {e}")
        print(f"   ⚠️  Training from scratch instead")
        
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
            tensorboard_log="./logs/stage3/",
            device="cpu",
            policy_kwargs=dict(
                net_arch=[256, 256, 128]
            )
        )
    
    # Setup callbacks
    progress_callback = ProgressCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/stage3_lower_intensity_checkpoints/",
        name_prefix="flip_recovery_policy",
        save_vecnormalize=True
    )
    
    callbacks = [progress_callback, checkpoint_callback]
    
    # Create directories
    Path("./models/stage3_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("./logs/stage3/").mkdir(parents=True, exist_ok=True)
    
    print("\n[3/3] Starting PPO training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Checkpoints: Every 25,000 steps (~50 episodes)")
    print(f"   Estimated time: {args.timesteps / 30000:.1f} hours (at ~30k steps/hour)")
    print()
    print("="*70)
    print("🚀 TRAINING STARTED")
    print("="*70)
    print("Watch for:")
    print("  - Recovery Rate: Should increase from 0% → 70%+")
    print("  - Recovery Time: Should decrease as learning improves")
    print("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        # Save final model
        model.save("./models/stage3_lower_intensity_checkpoints/flip_recovery_policy")
        env.save("./models/stage3_lower_intensity_checkpoints/flip_recovery_vecnormalize.pkl")
        
        print(f"\n💾 Model saved:")
        print(f"   - ./models/stage3_lower_intensity_checkpoints/flip_recovery_policy.zip")
        print(f"   - ./models/stage3_lower_intensity_checkpoints/flip_recovery_vecnormalize.pkl")
        
        print(f"\n📊 Training Statistics:")
        print(f"   Total episodes: {progress_callback.episode_count}")
        if progress_callback.episode_rewards:
            print(f"   Avg return: {np.mean(progress_callback.episode_rewards[-50:]):.1f} (last 50)")
            print(f"   Recovery rate: {np.mean(progress_callback.recovery_successes[-50:])*100:.0f}% (last 50)")
            if len(progress_callback.recovery_times) > 0:
                print(f"   Avg recovery time: {np.mean(progress_callback.recovery_times[-20:]):.0f} steps")
        
        print("\n✅ Next step: Run test_stage3_policy_v2.py to evaluate!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save("./models/stage3_lower_intensity_checkpoints/flip_recovery_policy_interrupted")
        env.save("./models/stage3_lower_intensity_checkpoints/flip_recovery_vecnormalize_interrupted.pkl")
        print("💾 Model saved at interruption point")
    
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stage2-model', type=str, 
                    default='./models/hover_disturbance_policy_interrupted.zip',
                    help='Path to Stage 2 trained policy')
    parser.add_argument('--timesteps', type=int, default=300000,
                        help='Total training timesteps')
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Maximum wind strength (m/s)')
    parser.add_argument('--flip-prob', type=float, default=1.0,
                        help='Probability of starting flipped (0.0-1.0)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (lower than Stage 2 for fine-tuning)')
    
    args = parser.parse_args()
    
    main(args)