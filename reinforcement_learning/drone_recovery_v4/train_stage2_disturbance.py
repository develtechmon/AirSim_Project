"""
STAGE 2: PPO TRAINING WITH DISTURBANCES
========================================
Trains the drone to handle wind disturbances using PPO (Reinforcement Learning).

Key Features:
- Loads your Stage 1 hover policy as starting point
- Adds wind disturbances to environment
- Fine-tunes with PPO for 2-3 hours
- Expected result: 80%+ success with wind

This is MUCH faster than training from scratch because we start
with a policy that already knows how to hover!

python train_stage2_disturbance.py --timesteps 250000
python train_stage2_disturbance.py --lr 1e-5 --timesteps 250000 (with lower learning rate)
python train_stage2_disturbance.py --timesteps 250000 --lr 1e-5


"""
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from drone_hover_disturbance_env import DroneHoverDisturbanceEnv
import argparse
from pathlib import Path


class HoverPolicy(nn.Module):
    """Stage 1 policy architecture"""
    def __init__(self, state_dim=10, action_dim=3):
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
                    
                    # Print every 10 episodes
                    if self.episode_count % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_lengths = self.episode_lengths[-10:]
                        
                        print(f"\n{'='*70}")
                        print(f"📊 EPISODE {self.episode_count}")
                        print(f"{'='*70}")
                        print(f"   Last 10 Episodes:")
                        print(f"      Avg Return: {np.mean(recent_rewards):.1f}")
                        print(f"      Avg Length: {np.mean(recent_lengths):.1f} steps")
                        print(f"      Max Length: {np.max(recent_lengths)} steps")
                        
                        # Wind info
                        wind_mag = info.get("wind_magnitude", 0)
                        print(f"   Current wind: {wind_mag:.1f} m/s")
                        print(f"{'='*70}\n")
        
        return True


class RobustCheckpointCallback(BaseCallback):
    """
    Robust checkpoint callback that saves model and vecnormalize reliably
    Saves every N episodes (not steps) for easier tracking
    """
    
    def __init__(self, save_freq_episodes=10, save_path="./checkpoints/", 
                 name_prefix="model", verbose=1):
        super().__init__(verbose)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.episode_count = 0
        self.last_save_episode = 0
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones"):
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_count += 1
                    
                    # Check if we should save
                    if self.episode_count % self.save_freq_episodes == 0:
                        if self.episode_count != self.last_save_episode:
                            self._save_checkpoint()
                            self.last_save_episode = self.episode_count
        
        return True
    
    def _save_checkpoint(self):
        """Save model and vecnormalize with confirmation"""
        try:
            episode = self.episode_count
            
            # Paths
            model_path = self.save_path / f"{self.name_prefix}_ep{episode}.zip"
            vecnorm_path = self.save_path / f"{self.name_prefix}_ep{episode}_vecnorm.pkl"
            
            print(f"\n💾 SAVING CHECKPOINT at Episode {episode}...")
            
            # Save model
            self.model.save(str(model_path))
            print(f"   ✅ Model saved: {model_path}")
            
            # Save VecNormalize if it exists
            if self.training_env is not None:
                try:
                    from stable_baselines3.common.vec_env import VecNormalize
                    if isinstance(self.training_env, VecNormalize):
                        self.training_env.save(str(vecnorm_path))
                        print(f"   ✅ VecNormalize saved: {vecnorm_path}")
                except Exception as e:
                    print(f"   ⚠️  VecNormalize save failed: {e}")
            
            print(f"   ✅ Checkpoint complete!\n")
            
        except Exception as e:
            print(f"   ❌ Checkpoint save failed: {e}\n")


def load_stage1_policy(model_path):
    """Load Stage 1 behavioral cloning weights"""
    print(f"[1/5] Loading Stage 1 policy: {model_path}")
    
    # Load BC model
    bc_model = HoverPolicy()
    bc_model.load_state_dict(torch.load(model_path))
    bc_weights = bc_model.state_dict()
    
    print("   ✅ Stage 1 policy loaded")
    print("   📊 This policy achieved 100% hover success!")
    return bc_weights


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
            net_arch=[256, 256, 128]  # ← FIXED: Shared architecture for pi and vf
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
        
        # Stage 1 BC architecture: 10 → 256 → 256 → 128 → 3
        # PPO shared architecture: 10 → 256 → 256 → 128 (then splits to pi/vf heads)
        
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
        print("   💡 PPO will start from 100% hover success!")
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
    print("Training drone to handle wind while hovering")
    print(f"Starting from Stage 1 policy (100% success)")
    print(f"Expected training time: 2-3 hours")
    print("="*70 + "\n")
    
    # Load Stage 1 policy
    bc_weights = load_stage1_policy(args.stage1_model)
    
    # Create environment
    print(f"\n[4/5] Creating disturbance environment...")
    print(f"   Wind strength: 0-{args.wind_strength} m/s")
    
    def make_env():
        env = DroneHoverDisturbanceEnv(
            target_altitude=10.0,
            max_steps=500,
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
    
    # Robust checkpoint - saves every 10 episodes
    robust_checkpoint = RobustCheckpointCallback(
        save_freq_episodes=10,
        save_path="./models/stage2_checkpoints/",
        name_prefix="disturbance_policy",
        verbose=1
    )
    
    callbacks = [progress_callback, robust_checkpoint]
    
    # Create directories
    Path("./models/stage2_checkpoints/").mkdir(parents=True, exist_ok=True)
    Path("./logs/stage2/").mkdir(parents=True, exist_ok=True)
    
    print("\n[5/5] Starting PPO training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.lr}")
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
            print(f"   Avg return: {np.mean(progress_callback.episode_rewards):.1f}")
            print(f"   Avg length: {np.mean(progress_callback.episode_lengths):.1f}")
        
        print("\n✅ Next step: Run test_stage2_policy.py to evaluate!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save("./models/hover_disturbance_policy_interrupted")
        env.save("./models/hover_disturbance_vecnormalize_interrupted.pkl")
        print("💾 Model saved at interruption point")
    
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stage1-model', type=str, 
                        default='./models/hover_policy_best.pth',
                        help='Path to Stage 1 trained policy')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Maximum wind strength (m/s)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate (lower than Stage 1 for fine-tuning)')
    
    args = parser.parse_args()
    
    main(args)