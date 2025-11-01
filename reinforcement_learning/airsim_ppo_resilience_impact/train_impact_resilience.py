"""
PPO Training Script for Impact Resilience
==========================================

Trains a PPO agent to recover from various impact types using
literature-based feature extraction and physics-realistic impact simulation.

Based on:
- Paper #8,9,10: PPO for quadcopter control and recovery
- Paper #1: IMU-based collision detection
- Paper #3,6,7: Feature extraction from IMU data

Impact Types Learned:
1. Sharp Collision - high jerk, brief duration
2. Sustained Force - low jerk, long duration  
3. Rotational - high angular acceleration
4. Free-fall - vertical acceleration drop

Recovery Strategies:
- Altitude maintenance
- Angular stabilization
- Position correction
- Velocity damping
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import json
import sys

sys.path.insert(0, '/home/claude')
from drone_impact_resilience_env import ImpactResilienceEnv


class ImpactStatsCallback(BaseCallback):
    """Track impact recovery statistics"""
    
    def __init__(self, check_freq=10, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_recoveries = []
        self.best_recovery_rate = 0.0
    
    def _on_step(self):
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            
            self.episode_recoveries.append(info.get('recovered', False))
            
            if self.episode_count % self.check_freq == 0:
                self._print_progress()
        
        return True
    
    def _print_progress(self):
        recent = self.episode_recoveries[-self.check_freq:]
        recovery_rate = np.mean(recent) * 100
        avg_reward = np.mean(self.episode_rewards[-self.check_freq:])
        
        self.best_recovery_rate = max(self.best_recovery_rate, recovery_rate)
        
        print(f"\n{'='*70}")
        print(f"📊 EPISODE {self.episode_count} | STEPS {self.num_timesteps}")
        print(f"  Recovery Rate: {recovery_rate:>6.1f}% (Best: {self.best_recovery_rate:.1f}%)")
        print(f"  Avg Reward:    {avg_reward:>6.1f}")
        print(f"{'='*70}")


def make_env():
    def _init():
        return Monitor(ImpactResilienceEnv())
    return _init


def train_ppo(total_timesteps=300000):
    """Train PPO for impact resilience"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"/home/claude/impact_ppo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("🚀 IMPACT RESILIENCE TRAINING")
    print(f"{'='*70}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Save dir:  {save_dir}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=save_dir,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=1
    )
    
    # Callbacks
    stats_cb = ImpactStatsCallback(check_freq=10)
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo"
    )
    
    print("🎓 Training started...")
    print("Expected milestones:")
    print("  30k:  50% recovery")
    print("  80k:  70% recovery")
    print("  150k: 85%+ recovery\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[stats_cb, checkpoint_cb],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
    
    # Save
    model.save(os.path.join(save_dir, "ppo_final"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete! Saved to {save_dir}")
    
    return model, env, save_dir


if __name__ == "__main__":
    print("\nIMPACT RESILIENCE - PPO TRAINING")
    print("="*70)
    
    timesteps = input("Training timesteps (default 300000): ").strip()
    timesteps = int(timesteps) if timesteps else 300000
    
    input("\nPress ENTER to start training...")
    
    train_ppo(total_timesteps=timesteps)