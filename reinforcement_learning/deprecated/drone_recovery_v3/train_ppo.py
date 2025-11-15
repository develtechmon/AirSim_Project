"""
PPO Training Script for Drone Recovery - FIXED VERSION
=======================================================
Trains drone to recover from impacts using curriculum learning.

KEY FIXES:
1. Success logic: crashes NEVER count as success
2. Proper episode length tracking
3. Real-time progress feedback
4. Reduced reward scale
"""

import argparse
import os
import time
import json
from datetime import datetime
from collections import deque
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from airsim_recovery_env import AirSimDroneRecoveryEnv


class ProgressBarCallback(BaseCallback):
    """Display progress bar during training with live metrics."""
    
    def __init__(self, total_timesteps: int, update_freq: int = 100):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.last_print_time = time.time()
        self.episode_count = 0
        self.last_mean_reward = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        
        # Update every N steps or when episode ends
        dones = self.locals.get("dones", [False])
        
        if self.n_calls % self.update_freq == 0 or dones[0]:
            current_time = time.time()
            elapsed = current_time - self.last_print_time
            
            if elapsed > 0:
                steps_per_sec = self.update_freq / elapsed if not dones[0] else 0
            else:
                steps_per_sec = 0
            
            progress = self.n_calls / self.total_timesteps
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Get recent reward if available
            infos = self.locals.get("infos", [{}])
            if dones[0] and "episode" in infos[0]:
                self.last_mean_reward = infos[0]["episode"]["r"]
                self.episode_count += 1
            
            print(f"\rStep {self.n_calls:>7}/{self.total_timesteps} [{bar}] {progress*100:>5.1f}% | "
                  f"{steps_per_sec:>5.1f} steps/s | Episodes: {self.episode_count:>4} | "
                  f"Last Reward: {self.last_mean_reward:>7.2f}", end="", flush=True)
            
            self.last_print_time = current_time
        
        return True


class DetailedLoggingCallback(BaseCallback):
    """Custom callback for detailed training monitoring."""
    
    def __init__(
        self, 
        check_freq: int = 1000,
        stage_advancement_threshold: float = 0.80,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.stage_advancement_threshold = stage_advancement_threshold
        
        # Metrics tracking
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_successes = []
        self.recent_returns = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        
        self.training_start_time = time.time()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        
        # Check if episode ended
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            # Extract episode info
            infos = self.locals.get("infos", [{}])
            info = infos[0]
            
            # Get episode statistics from VecMonitor wrapper
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)
                self.recent_returns.append(ep_return)
                self.recent_lengths.append(ep_length)
                
                # FIXED: Success logic based on termination reason
                reason = info.get("reason", "timeout")
                
                # Define success based on termination reason AND performance
                if reason == "hover_success":
                    # Explicit success condition (Stage 1)
                    is_success = True
                elif reason in ["ground_collision", "too_low", "out_of_bounds"]:
                    # Explicit failure conditions - NEVER count as success
                    is_success = False
                elif reason == "timeout":
                    # Timeout can be success if performance was good
                    # Good performance = positive return AND reasonable episode length
                    is_success = (ep_return > 0) and (ep_length > 100)
                else:
                    # Unknown reason - conservative classification
                    is_success = (ep_return > 50)
                
                self.episode_successes.append(1 if is_success else 0)
                self.recent_successes.append(1 if is_success else 0)
                
                # Log to TensorBoard
                self.logger.record("episode/return", ep_return)
                self.logger.record("episode/length", ep_length)
                self.logger.record("episode/success", int(is_success))
                
                # Console output for immediate feedback
                ep_num = len(self.episode_returns)
                print(f"\n{'='*80}")
                print(f"📊 EPISODE {ep_num} COMPLETE")
                print(f"{'='*80}")
                print(f"   Return: {ep_return:>8.2f}")
                print(f"   Length: {ep_length:>8.0f} steps")
                print(f"   Reason: {reason}")
                print(f"   Success: {'✅ YES' if is_success else '❌ NO'}")
                if len(self.recent_returns) >= 10:
                    print(f"   Rolling Avg (last 10): {np.mean(list(self.recent_returns)[-10:]):.2f}")
                    print(f"   Rolling Length (last 10): {np.mean(list(self.recent_lengths)[-10:]):.0f} steps")
                print(f"{'='*80}\n")
        
        # Periodic detailed logging
        if self.n_calls % self.check_freq == 0 and self.n_calls > 0:
            self._detailed_log()
        
        return True
    
    def _detailed_log(self):
        """Log detailed training statistics."""
        if len(self.episode_returns) == 0:
            return
        
        # Compute statistics
        mean_return = np.mean(self.recent_returns)
        mean_length = np.mean(self.recent_lengths)
        success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
        
        # Time statistics
        elapsed = time.time() - self.training_start_time
        episodes_per_hour = len(self.episode_returns) / (elapsed / 3600)
        
        print(f"\n{'='*80}")
        print(f"📈 TRAINING STATISTICS (Step {self.n_calls})")
        print(f"{'='*80}")
        print(f"   Episodes: {len(self.episode_returns)}")
        print(f"   Mean Return (last 100): {mean_return:.2f}")
        print(f"   Mean Length (last 100): {mean_length:.0f} steps")
        print(f"   Success Rate (last 100): {success_rate:.1%}")
        print(f"   Episodes/Hour: {episodes_per_hour:.1f}")
        print(f"   Training Time: {elapsed/3600:.2f} hours")
        print(f"{'='*80}\n")
        
        # Log to TensorBoard
        self.logger.record("train/mean_return_100", mean_return)
        self.logger.record("train/mean_length_100", mean_length)
        self.logger.record("train/success_rate_100", success_rate)


def train_drone_recovery(
    total_timesteps: int = 200_000,
    stage: int = 1,
    model_path: Optional[str] = None,
    log_dir: str = "./logs",
    save_freq: int = 10_000,
    debug: bool = False
) -> PPO:
    """
    Train PPO agent for drone impact recovery.
    
    Args:
        total_timesteps: Total training steps
        stage: Starting curriculum stage (1, 2, or 3)
        model_path: Path to pretrained model (optional)
        log_dir: Directory for logs and checkpoints
        save_freq: Save model every N steps
        debug: Enable debug logging
    
    Returns:
        Trained PPO model
    """
    
    print("\n" + "="*80)
    print("🚁 DRONE IMPACT RECOVERY TRAINING")
    print("="*80)
    print(f"Stage: {stage}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Debug Mode: {debug}")
    print(f"Log Directory: {log_dir}")
    print("="*80 + "\n")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"PPO_stage{stage}_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print(f"📁 Logs will be saved to: {run_log_dir}")
    print(f"📁 Models will be saved to: ./models/")
    print()
    
    # Create environment with proper monitoring wrapper
    print("🔧 Initializing AirSim environment...")
    
    def make_env():
        env = AirSimDroneRecoveryEnv(stage=stage, debug=debug)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # CRITICAL: VecMonitor must wrap the vec env to track episodes
    env = VecMonitor(env, filename=os.path.join(run_log_dir, "monitor.csv"))
    
    print("✅ Environment created and wrapped with VecMonitor")
    print(f"   Episode data will be logged to: {os.path.join(run_log_dir, 'monitor.csv')}")
    print()
    
    # Configure PPO hyperparameters
    ppo_config = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "n_steps": 2048,  # Rollout length
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": run_log_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"🤖 PPO Configuration:")
    for key, value in ppo_config.items():
        if key not in ["policy", "env", "tensorboard_log"]:
            print(f"   {key}: {value}")
    print(f"   Device: {ppo_config['device']}")
    print()
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"📂 Loading pretrained model from: {model_path}")
        model = PPO.load(model_path, env=env, **{k: v for k, v in ppo_config.items() if k not in ["policy", "env", "tensorboard_log"]})
    else:
        print("🆕 Creating new PPO model...")
        model = PPO(**ppo_config)
    
    print("✅ Model created")
    print()
    
    # Setup callbacks
    progress_callback = ProgressBarCallback(
        total_timesteps=total_timesteps,
        update_freq=100
    )
    
    logging_callback = DetailedLoggingCallback(
        check_freq=1000,
        stage_advancement_threshold=0.80,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/checkpoints",
        name_prefix=f"ppo_drone_stage{stage}",
        verbose=1
    )
    
    print("📊 Callbacks configured:")
    print(f"   Progress bar: Updates every 100 steps")
    print(f"   Detailed logging: Every 1000 steps")
    print(f"   Checkpoints: Every {save_freq:,} steps")
    print()
    
    # Configure logger - CRITICAL for CSV and TensorBoard
    new_logger = configure(run_log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    print("✅ Logger configured:")
    print(f"   CSV: {os.path.join(run_log_dir, 'progress.csv')}")
    print(f"   TensorBoard: {run_log_dir}")
    print()
    
    # Start training
    print(f"🎯 Starting training for {total_timesteps:,} steps...")
    print(f"💡 Monitor progress:")
    print(f"   TensorBoard: tensorboard --logdir={log_dir}")
    print()
    print("="*80)
    print("🚀 TRAINING STARTED - Watch for episode completions below")
    print("="*80 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, logging_callback, checkpoint_callback],
            progress_bar=False  # Using custom progress bar
        )
        
        print("\n\n✅ Training completed successfully!")
        
        # Save final model
        final_save_path = f"models/ppo_drone_stage{stage}_final_{timestamp}.zip"
        model.save(final_save_path)
        print(f"💾 Final model saved: {final_save_path}")
        
        # Save training metadata
        metadata = {
            "stage": stage,
            "total_timesteps": total_timesteps,
            "final_mean_return": float(np.mean(logging_callback.recent_returns)) if logging_callback.recent_returns else 0,
            "final_success_rate": float(np.mean(logging_callback.recent_successes)) if logging_callback.recent_successes else 0,
            "training_time_hours": (time.time() - logging_callback.training_start_time) / 3600,
            "timestamp": timestamp,
            "log_dir": run_log_dir
        }
        
        metadata_path = f"models/metadata_stage{stage}_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Training metadata saved: {metadata_path}")
        
        print(f"\n📊 View results:")
        print(f"   TensorBoard: tensorboard --logdir={run_log_dir}")
        print(f"   CSV: {os.path.join(run_log_dir, 'progress.csv')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        save_path = f"models/ppo_drone_stage{stage}_interrupted_{timestamp}.zip"
        model.save(save_path)
        print(f"💾 Model saved: {save_path}")
    
    finally:
        env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train drone recovery with PPO")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Training stage")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps")
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    model = train_drone_recovery(
        total_timesteps=args.timesteps,
        stage=args.stage,
        model_path=args.model,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        debug=args.debug
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())