"""
PPO TRAINING - PRODUCTION READY
Matches recovery_env_PRODUCTION.py environment
Clean, simple, and works with proven parameters
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from recovery_env import PostImpactRecoveryEnv


class TrainingMetricsCallback(BaseCallback):
    """Track training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.crashes = []
        
    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if self.locals.get('dones', [False])[0]:
                # Episode ended
                ep_info = info.get('episode', {})
                ep_reward = ep_info.get('r', 0)
                ep_length = ep_info.get('l', 0)
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Track success/crash
                if info.get('success', False):
                    self.successes.append(1)
                else:
                    self.crashes.append(1)
                
                # Print every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    self._print_progress()
        
        return True
    
    def _print_progress(self):
        total = len(self.episode_rewards)
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) > 100 else self.episode_rewards
        success_rate = (sum(self.successes) / total * 100) if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"📊 Episodes: {total} | Steps: {self.num_timesteps:,}")
        print(f"   Avg Reward: {np.mean(recent_rewards):.2f}")
        print(f"   Avg Length: {np.mean(self.episode_lengths[-100:]):.1f} steps")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Crashes: {len(self.crashes)}")
        print(f"{'='*70}")


def create_env(stage: str, enable_logging: bool = False):
    """Create training environment"""
    env = PostImpactRecoveryEnv(
        training_stage=stage,
        max_episode_steps=500,
        spawn_altitude=15.0,
        enable_logging=enable_logging
    )
    env = Monitor(env)
    return env


def train_stage(
    stage: str,
    total_timesteps: int,
    previous_model_path: Optional[str] = None,
    save_dir: str = "./models",
    log_dir: str = "./logs",
    device: str = "auto"
):
    """
    Train a single stage
    
    Args:
        stage: "hover", "disturbance", or "impact"
        total_timesteps: Training steps
        previous_model_path: Load weights from previous stage
        save_dir: Model save directory
        log_dir: Log directory
        device: "auto", "cuda", or "cpu"
    """
    print("\n" + "="*70)
    print(f"🚀 TRAINING STAGE: {stage.upper()}")
    print("="*70)
    print(f"Timesteps: {total_timesteps:,}")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    def make_env():
        return create_env(stage, enable_logging=False)
    
    env = DummyVecEnv([make_env])
    
    # Optional: Add normalization (can disable if causing issues)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995
    )
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"stage_{stage}_{timestamp}")
    tensorboard_log = configure(log_path, ["tensorboard", "stdout"])
    
    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"✓ Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load previous model or create new
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"✓ Loading previous model: {previous_model_path}")
        model = PPO.load(previous_model_path, env=env)
        model.learning_rate = 5e-5  # Lower LR for fine-tuning
    else:
        print("✓ Creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
            tensorboard_log=log_path,
            policy_kwargs=dict(net_arch=[256, 256])
        )
    
    model.set_logger(tensorboard_log)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, f"stage_{stage}"),
        name_prefix=f"ppo_{stage}",
        save_vecnormalize=True
    )
    
    metrics_callback = TrainingMetricsCallback()
    callbacks = CallbackList([checkpoint_callback, metrics_callback])
    
    print(f"\n🎯 TRAINING STARTED")
    print(f"TensorBoard: tensorboard --logdir {log_path}")
    print("="*70 + "\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        print(f"\n✅ Stage {stage.upper()} completed")
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted")
    
    # Save final model
    final_path = os.path.join(save_dir, f"ppo_{stage}_final")
    model.save(final_path)
    env.save(os.path.join(save_dir, f"vec_normalize_{stage}.pkl"))
    print(f"💾 Saved: {final_path}.zip\n")
    
    env.close()
    return final_path + ".zip"


def train_full_curriculum(
    hover_steps: int = 50000,
    disturbance_steps: int = 100000,
    impact_steps: int = 200000,
    save_dir: str = "./models",
    log_dir: str = "./logs"
):
    """
    Full 3-stage curriculum training
    """
    print("\n" + "="*70)
    print("🎓 CURRICULUM TRAINING PIPELINE")
    print("="*70)
    print(f"Stage 1 (HOVER):       {hover_steps:,} steps")
    print(f"Stage 2 (DISTURBANCE): {disturbance_steps:,} steps")
    print(f"Stage 3 (IMPACT):      {impact_steps:,} steps")
    print(f"Total:                 {hover_steps + disturbance_steps + impact_steps:,} steps")
    print("="*70)
    
    # Stage 1: Hover
    print("\n📍 Stage 1/3: Learning stable hover...")
    hover_model = train_stage(
        stage="hover",
        total_timesteps=hover_steps,
        save_dir=save_dir,
        log_dir=log_dir
    )
    
    input("\n⏸️  Press Enter to continue to Stage 2...")
    
    # Stage 2: Disturbance
    print("\n📍 Stage 2/3: Learning disturbance recovery...")
    disturbance_model = train_stage(
        stage="disturbance",
        total_timesteps=disturbance_steps,
        previous_model_path=hover_model,
        save_dir=save_dir,
        log_dir=log_dir
    )
    
    input("\n⏸️  Press Enter to continue to Stage 3...")
    
    # Stage 3: Impact
    print("\n📍 Stage 3/3: Learning full impact recovery...")
    impact_model = train_stage(
        stage="impact",
        total_timesteps=impact_steps,
        previous_model_path=disturbance_model,
        save_dir=save_dir,
        log_dir=log_dir
    )
    
    print("\n" + "="*70)
    print("🎉 CURRICULUM TRAINING COMPLETE!")
    print("="*70)
    print(f"Final model: {impact_model}")
    print("="*70 + "\n")
    
    return impact_model


def test_model(
    model_path: str,
    vec_normalize_path: str,
    stage: str = "impact",
    num_episodes: int = 10
):
    """
    Test trained model
    """
    print("\n" + "="*70)
    print("🧪 MODEL TESTING")
    print("="*70)
    
    env = create_env(stage, enable_logging=True)
    env = DummyVecEnv([lambda: env])
    
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded normalization")
    
    model = PPO.load(model_path)
    print("✓ Model loaded\n")
    
    results = {
        'rewards': [],
        'lengths': [],
        'successes': []
    }
    
    for episode in range(num_episodes):
        print(f"📍 Episode {episode+1}/{num_episodes}")
        obs = env.reset()
        ep_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            
            if done[0]:
                info = info[0]
                results['rewards'].append(ep_reward)
                results['lengths'].append(info.get('step', 0))
                results['successes'].append(info.get('success', False))
                
                print(f"  {'✓ Success' if info.get('success') else '✗ Failed'} | Reward: {ep_reward:.2f}\n")
                break
    
    env.close()
    
    # Summary
    print("="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Success Rate: {np.mean(results['successes'])*100:.1f}%")
    print(f"Avg Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Avg Length: {np.mean(results['lengths']):.1f} steps")
    print("="*70 + "\n")


def interactive_menu():
    """Interactive training menu"""
    print("\n" + "="*70)
    print("🚁 POST-IMPACT RECOVERY - PPO TRAINING")
    print("="*70)
    print("\n1. Full curriculum (hover → disturbance → impact)")
    print("2. Train single stage")
    print("3. Test trained model")
    print("4. Exit")
    print("="*70)
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        # Full curriculum
        hover = int(input("Hover steps (default 50000): ") or "50000")
        disturbance = int(input("Disturbance steps (default 100000): ") or "100000")
        impact = int(input("Impact steps (default 200000): ") or "200000")
        
        train_full_curriculum(hover, disturbance, impact)
        
    elif choice == "2":
        # Single stage
        stage = input("Stage (hover/disturbance/impact): ").strip().lower()
        if stage not in ["hover", "disturbance", "impact"]:
            print("❌ Invalid stage")
            return interactive_menu()
        
        timesteps = int(input("Timesteps: "))
        prev_model = input("Previous model path (Enter to skip): ").strip() or None
        
        train_stage(stage, timesteps, prev_model)
        
    elif choice == "3":
        # Test model
        model_path = input("Model path (.zip): ").strip()
        vec_path = input("Vec normalize path (.pkl): ").strip()
        stage = input("Test stage (hover/disturbance/impact): ").strip().lower()
        episodes = int(input("Episodes (default 10): ") or "10")
        
        test_model(model_path, vec_path, stage, episodes)
        
    elif choice == "4":
        print("\n👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("❌ Invalid choice")
    
    # Loop back
    interactive_menu()


if __name__ == "__main__":
    print("\n🔍 System Check:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    interactive_menu()