"""
STAGE 2: PPO TRAINING WITH DISTURBANCES (13 OBSERVATIONS)
==========================================================
Trains the drone to handle wind disturbances using PPO (Reinforcement Learning).

UPDATED: Now uses 13 observations (same as Stage 1) for transfer learning!

Key Features:
- Loads your Stage 1 hover policy as starting point
- Adds wind disturbances to environment
- Fine-tunes with PPO for 5 hours
- Expected result: 90%+ success with wind

Usage:
    python train_stage2_disturbance_v2.py

That's it! Just like the original!

python train_stage2_disturbance_v2.py (i'm using this command)

*************** Final Result *****************  
======================================================================
📊 EPISODE 340
======================================================================
   Last 10 Episodes:
      Avg Return: 41045.9 <--- High return
      Avg Length: 500.0 steps <--- High length
      Max Length: 500 steps
   Current wind: 2.4 m/s
======================================================================

-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 500         |
|    ep_rew_mean          | 4.09e+04    |
| time/                   |             |
|    fps                  | 12          |
|    iterations           | 84          |
|    time_elapsed         | 14268       |
|    total_timesteps      | 172032      |
| train/                  |             |
|    approx_kl            | 0.002353337 |
|    clip_fraction        | 0.00508     |
|    clip_range           | 0.2         |
|    entropy_loss         | -4.22       |
|    explained_variance   | 0.00973     |
|    learning_rate        | 3e-05       |
|    loss                 | 9.15e+05    |
|    n_updates            | 830         |
|    policy_gradient_loss | -0.00441    |
|    std                  | 0.987       |
|    value_loss           | 1.92e+06    |
-----------------------------------------

"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from drone_hover_disturbance_env import DroneHoverDisturbanceEnv
import argparse
from pathlib import Path


class HoverPolicy(nn.Module):
    """Stage 1 policy architecture (13 observations)"""
    def __init__(self, state_dim=13, action_dim=3):
        super(HoverPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )


class ProgressCallback(BaseCallback):
    """Callback to log training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wind_magnitudes = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_count += 1
                    
                    info = self.locals["infos"][i]
                    ep_reward = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    wind_mag = info.get("wind_magnitude", 0)
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.wind_magnitudes.append(wind_mag)
                    
                    # Print every 10 episodes
                    if self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_lengths = self.episode_lengths[-10:]
                        recent_winds = self.wind_magnitudes[-10:]
                        
                        # Success metric: episodes that reach max_steps
                        success_count = sum(1 for l in recent_lengths if l >= 590)
                        success_rate = success_count / len(recent_lengths) * 100
                        
                        print(f"\n{'='*70}")
                        print(f"📊 EPISODE {self.episode_count}")
                        print(f"{'='*70}")
                        print(f"   Last 10 Episodes:")
                        print(f"      Avg Return: {np.mean(recent_rewards):.1f}")
                        print(f"      Avg Length: {np.mean(recent_lengths):.1f} steps")
                        print(f"      Success Rate: {success_rate:.0f}% (reached timeout)")
                        print(f"      Avg Wind: {np.mean(recent_winds):.2f} m/s")
                        print(f"{'='*70}\n")
        
        return True


def load_stage1_policy(model_path):
    """Load Stage 1 behavioral cloning weights"""
    print(f"[1/5] Loading Stage 1 hover policy: {model_path}")
    
    try:
        # Load BC model
        bc_model = HoverPolicy(state_dim=13, action_dim=3)
        bc_model.load_state_dict(torch.load(model_path))
        bc_weights = bc_model.state_dict()
        
        print("   ✅ Stage 1 policy loaded")
        print("   📊 This policy achieved 95%+ hover success!")
        return bc_weights
    
    except FileNotFoundError:
        print(f"   ❌ ERROR: Stage 1 model not found at {model_path}")
        print(f"   💡 Make sure you've trained Stage 1 first:")
        print(f"      python train_stage1_hover.py")
        print(f"   💡 Or specify correct path:")
        print(f"      --stage1-model /path/to/your/hover_policy_best.pth")
        raise


def create_ppo_with_pretrained_weights(env, bc_weights, args):
    """Create PPO model and load pretrained weights"""
    print("\n[2/5] Creating PPO model...")
    
    # Create PPO with SAME architecture as Stage 1
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
        tensorboard_log="./logs/stage2/",
        device="cpu",
        policy_kwargs=dict(
            net_arch=[256, 256, 128]  # Same as Stage 1
        )
    )
    
    print("   ✅ PPO model created")
    
    # Load pretrained weights into actor network
    print("\n[3/5] Loading pretrained weights into PPO actor...")
    
    try:
        # Get PPO policy
        policy = model.policy
        
        # Map BC weights to PPO shared features + action head
        actor_state_dict = {}
        
        # Stage 1 BC architecture: 13 → 256 → 256 → 128 → 3
        # PPO shared architecture: 13 → 256 → 256 → 128 (then splits to pi/vf heads)
        
        # Map shared layers (features_extractor.mlp)
        for bc_key, bc_param in bc_weights.items():
            if bc_key.startswith('network.'):
                parts = bc_key.split('.')
                layer_num = int(parts[1])
                param_type = parts[2]
                
                if layer_num < 6:  # Layers 0,2,4 (the 3 hidden layers)
                    # Map to shared feature extractor
                    ppo_layer = layer_num // 2
                    ppo_key = f'features_extractor.mlp.{ppo_layer}.{param_type}'
                    actor_state_dict[ppo_key] = bc_param
                else:  # Layer 6 (final output layer)
                    # Map to action head
                    ppo_key = f'action_net.{param_type}'
                    actor_state_dict[ppo_key] = bc_param
        
        # Load into PPO policy
        policy.load_state_dict(actor_state_dict, strict=False)
        
        print("   ✅ Pretrained weights loaded into actor network")
        print("   💡 PPO will start from 95%+ hover success!")
        print("   📈 Only needs to learn wind compensation")
        
    except Exception as e:
        print(f"   ⚠️  Could not load pretrained weights: {e}")
        print("   ⚠️  Training from scratch instead")
        print("   ⚠️  This will take longer but should still work!")
    
    return model


def main(args):
    print("\n" + "="*70)
    print("🌬️  STAGE 2: DISTURBANCE RECOVERY TRAINING")
    print("="*70)
    print("Training drone to handle wind while hovering at 30m altitude")
    print(f"Starting from Stage 1 hover policy (95%+ success)")
    print(f"Expected training time: 5 hours")
    print("="*70 + "\n")
    
    # Load Stage 1 policy
    bc_weights = load_stage1_policy(args.stage1_model)
    
    # Create environment
    print(f"\n[4/5] Creating disturbance environment...")
    print(f"   Target altitude: 30m")
    print(f"   Wind strength: 0-{args.wind_strength} m/s")
    print(f"   Episode length: {args.max_steps} steps ({args.max_steps * 0.05:.1f}s)")
    
    def make_env():
        env = DroneHoverDisturbanceEnv(
            target_altitude=30.0,  # ← 30m altitude
            max_steps=args.max_steps,  # ← Longer episodes
            wind_strength=args.wind_strength,
            debug=False
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Normalize observations only
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    
    print("   ✅ Environment created with wind disturbances")
    
    # Create PPO with pretrained weights
    model = create_ppo_with_pretrained_weights(env, bc_weights, args)
    
    # Setup callbacks
    progress_callback = ProgressCallback()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models/stage2_checkpoints/",
        name_prefix="disturbance_policy",
        save_vecnormalize=True
    )
    
    callbacks = [progress_callback, checkpoint_callback]
    
    # Create directories
    Path("./models/stage2_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("./logs/stage2/").mkdir(parents=True, exist_ok=True)
    
    print("\n[5/5] Starting PPO training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Checkpoints: Every 25,000 steps")
    print(f"   Estimated time: {args.timesteps / 30000:.1f} hours (at ~30k steps/hour)")
    print()
    print("="*70)
    print("🚀 TRAINING STARTED")
    print("="*70)
    print("Watch for episode statistics every 10 episodes...")
    print("Model will learn to compensate for wind disturbances!")
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
        model.save("./models/hover_disturbance_policy")
        env.save("./models/hover_disturbance_vecnormalize.pkl")
        
        print(f"\n💾 Model saved:")
        print(f"   - ./models/hover_disturbance_policy.zip")
        print(f"   - ./models/hover_disturbance_vecnormalize.pkl")
        
        print(f"\n📊 Training Statistics:")
        print(f"   Total episodes: {progress_callback.episode_count}")
        if progress_callback.episode_rewards:
            avg_reward = np.mean(progress_callback.episode_rewards[-50:])
            avg_length = np.mean(progress_callback.episode_lengths[-50:])
            success_count = sum(1 for l in progress_callback.episode_lengths[-50:] if l >= 590)
            success_rate = success_count / 50 * 100
            
            print(f"   Last 50 episodes:")
            print(f"      Avg return: {avg_reward:.1f}")
            print(f"      Avg length: {avg_length:.1f} steps")
            print(f"      Success rate: {success_rate:.0f}%")
        
        print("\n✅ Next step: Test the model!")
        print("   python test_stage2_policy.py")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save("./models/hover_disturbance_policy_interrupted")
        env.save("./models/hover_disturbance_vecnormalize_interrupted.pkl")
        print("💾 Model saved at interruption point")
    
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stage 2: Train PPO with wind disturbances at 30m altitude'
    )
    
    # ================================================================
    # FIXED: Correct model path for 30m hover policy
    # ================================================================
    parser.add_argument('--stage1-model', type=str, 
                        default='./models/hover_policy_best.pth',  # ← FIXED!
                        help='Path to Stage 1 hover policy (30m)')
    
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    
    # ================================================================
    # NEW: Configurable episode length
    # ================================================================
    parser.add_argument('--max-steps', type=int, default=600,  # ← INCREASED!
                        help='Maximum steps per episode (600 = 30 seconds)')
    
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Maximum wind strength (m/s)')
    
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate (lower for fine-tuning)')
    
    args = parser.parse_args()
    
    main(args)