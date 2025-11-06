"""
PPO Training Script for Drone Impact Recovery
==============================================
Features:
- Curriculum learning with automatic stage advancement
- Real-time training metrics and visualization
- Learning verification and convergence detection
- Automatic checkpointing and model saving
- TensorBoard integration
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from airsim_recovery_env import AirSimDroneRecoveryEnv


class DetailedLoggingCallback(BaseCallback):
    """
    Custom callback for detailed training monitoring and curriculum advancement.
    
    Tracks:
    - Episode returns and success rates
    - Learning progress metrics
    - Stage advancement criteria
    """
    
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
        self.current_stage = 1
        
        # Rolling statistics (last 100 episodes)
        self.recent_returns = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        
        # Training start time
        self.training_start_time = time.time()
        self.last_log_time = time.time()
    
    def _on_step(self) -> bool:
        """Called at each step."""
        
        # Check if episode ended
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            # Extract episode info
            infos = self.locals.get("infos", [{}])
            info = infos[0]
            
            # Get episode statistics
            if "episode" in info:
                ep_return = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)
                self.recent_returns.append(ep_return)
                
                # Check if episode was successful
                reason = info.get("reason", "timeout")
                is_success = (reason == "hover_success") or (ep_return > 50)
                self.episode_successes.append(1 if is_success else 0)
                self.recent_successes.append(1 if is_success else 0)
                
                # Log to TensorBoard
                self.logger.record("episode/return", ep_return)
                self.logger.record("episode/length", ep_length)
                self.logger.record("episode/success", int(is_success))
                self.logger.record("episode/reason", reason)
        
        # Periodic detailed logging
        if self.n_calls % self.check_freq == 0:
            self._detailed_log()
        
        return True
    
    def _detailed_log(self) -> None:
        """Comprehensive logging of training progress."""
        
        if len(self.recent_returns) == 0:
            return
        
        # Calculate statistics
        mean_return = np.mean(self.recent_returns)
        std_return = np.std(self.recent_returns)
        min_return = np.min(self.recent_returns)
        max_return = np.max(self.recent_returns)
        success_rate = np.mean(self.recent_successes) if len(self.recent_successes) > 0 else 0.0
        
        # Time statistics
        elapsed_time = time.time() - self.training_start_time
        time_since_last_log = time.time() - self.last_log_time
        steps_per_second = self.check_freq / time_since_last_log if time_since_last_log > 0 else 0
        
        # Log to TensorBoard
        self.logger.record("rollout/mean_return", mean_return)
        self.logger.record("rollout/std_return", std_return)
        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/stage", self.current_stage)
        self.logger.record("time/steps_per_second", steps_per_second)
        self.logger.record("time/elapsed_hours", elapsed_time / 3600)
        
        # Console output
        print("\n" + "="*80)
        print(f"📊 TRAINING PROGRESS - Step {self.n_calls:,} | Stage {self.current_stage}")
        print("="*80)
        print(f"📈 Episode Return (last 100 episodes):")
        print(f"   Mean: {mean_return:>8.2f} ± {std_return:.2f}")
        print(f"   Range: [{min_return:>7.2f}, {max_return:>7.2f}]")
        print(f"✅ Success Rate: {success_rate:>6.1%} ({int(success_rate*100)}/100 episodes)")
        print(f"⏱️  Performance: {steps_per_second:.1f} steps/sec | {elapsed_time/3600:.2f} hours elapsed")
        
        # Stage advancement check
        if self.current_stage < 3 and success_rate >= self.stage_advancement_threshold:
            print(f"\n🎉 STAGE {self.current_stage} MASTERED!")
            print(f"   Success rate: {success_rate:.1%} >= {self.stage_advancement_threshold:.1%}")
            print(f"   Advancing to Stage {self.current_stage + 1}...")
            self._advance_stage()
        
        # Check if learning is happening
        if len(self.episode_returns) >= 100:
            recent_100 = self.episode_returns[-100:]
            older_100 = self.episode_returns[-200:-100] if len(self.episode_returns) >= 200 else self.episode_returns[:100]
            improvement = np.mean(recent_100) - np.mean(older_100)
            
            if improvement > 0:
                print(f"📈 Learning Progress: +{improvement:.2f} reward improvement (last 100 vs previous 100)")
            else:
                print(f"⚠️  Warning: Reward decreased by {abs(improvement):.2f} - possible instability")
        
        print("="*80 + "\n")
        
        # Save best model
        if mean_return > self.best_mean_reward:
            self.best_mean_reward = mean_return
            save_path = f"models/best_model_stage{self.current_stage}.zip"
            self.model.save(save_path)
            print(f"💾 New best model saved: {save_path} (mean return: {mean_return:.2f})\n")
        
        self.last_log_time = time.time()
    
    def _advance_stage(self) -> None:
        """Advance to next curriculum stage."""
        self.current_stage += 1
        
        # Update environment stage
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'set_stage'):
                    env.set_stage(self.current_stage)
        
        # Reset statistics for new stage
        self.recent_successes.clear()
        self.recent_returns.clear()
        
        print(f"\n{'='*80}")
        print(f"🚀 CURRICULUM ADVANCEMENT")
        print(f"{'='*80}")
        print(f"   New Stage: {self.current_stage}")
        print(f"   New Objective: {self._get_stage_objective(self.current_stage)}")
        print(f"{'='*80}\n")
    
    def _get_stage_objective(self, stage: int) -> str:
        """Get objective description for stage."""
        objectives = {
            1: "Learn stable hover at target position",
            2: "Maintain stability under wind gusts and small disturbances",
            3: "Recover from severe impacts, flips, and collisions"
        }
        return objectives.get(stage, "Unknown")


class CurriculumManager:
    """Manages curriculum learning progression."""
    
    def __init__(self, stages: List[int] = [1, 2, 3]):
        self.stages = stages
        self.current_stage_idx = 0
        self.stage_histories = {stage: [] for stage in stages}
    
    def should_advance(self, success_rate: float, threshold: float = 0.80) -> bool:
        """Check if should advance to next stage."""
        return success_rate >= threshold and self.current_stage_idx < len(self.stages) - 1
    
    def advance(self) -> int:
        """Advance to next stage."""
        self.current_stage_idx += 1
        return self.stages[self.current_stage_idx]
    
    def get_current_stage(self) -> int:
        """Get current stage number."""
        return self.stages[self.current_stage_idx]


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
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    print("🔧 Initializing AirSim environment...")
    env = AirSimDroneRecoveryEnv(stage=stage, debug=debug)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    
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
        "tensorboard_log": log_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"\n🤖 PPO Configuration:")
    for key, value in ppo_config.items():
        if key not in ["policy", "env", "tensorboard_log"]:
            print(f"   {key}: {value}")
    print(f"   Device: {ppo_config['device']}")
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"\n📂 Loading pretrained model from: {model_path}")
        model = PPO.load(model_path, env=env, **{k: v for k, v in ppo_config.items() if k not in ["policy", "env"]})
    else:
        print("\n🆕 Creating new PPO model...")
        model = PPO(**ppo_config)
    
    # Setup callbacks
    logging_callback = DetailedLoggingCallback(
        check_freq=1000,
        stage_advancement_threshold=0.80,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/checkpoints",
        name_prefix=f"ppo_drone_stage{stage}"
    )
    
    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Start training
    print(f"\n🎯 Starting training for {total_timesteps:,} steps...")
    print(f"💡 Tip: Monitor progress in TensorBoard:")
    print(f"   tensorboard --logdir={log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[logging_callback, checkpoint_callback],
            progress_bar=True
        )
        
        print("\n✅ Training completed successfully!")
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "timestamp": timestamp
        }
        
        metadata_path = f"models/metadata_stage{stage}_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Training metadata saved: {metadata_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        save_path = f"models/ppo_drone_stage{stage}_interrupted.zip"
        model.save(save_path)
        print(f"💾 Model saved: {save_path}")
    
    finally:
        env.close()
    
    return model


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    stage: int = 1,
    render: bool = False
) -> Dict:
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to trained model
        n_episodes: Number of evaluation episodes
        stage: Curriculum stage
        render: Whether to render (not supported in AirSim)
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n🔍 Evaluating model: {model_path}")
    print(f"Episodes: {n_episodes} | Stage: {stage}\n")
    
    # Load model
    env = AirSimDroneRecoveryEnv(stage=stage, debug=True)
    model = PPO.load(model_path)
    
    episode_returns = []
    episode_lengths = []
    success_count = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            
            if ep_length % 100 == 0:
                print(f"  Step {ep_length}: Return={ep_return:.2f}, Pos Error={info['pos_error']:.2f}m")
        
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        
        if info.get("reason") == "hover_success":
            success_count += 1
        
        print(f"\nEpisode Result:")
        print(f"  Return: {ep_return:.2f}")
        print(f"  Length: {ep_length} steps")
        print(f"  Reason: {info.get('reason', 'unknown')}")
    
    env.close()
    
    # Compute statistics
    results = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": success_count / n_episodes
    }
    
    print(f"\n{'='*60}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Mean Length: {results['mean_length']:.1f} steps")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for drone impact recovery")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Curriculum stage")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.eval:
        if args.model is None:
            print("❌ Error: --model required for evaluation mode")
            exit(1)
        evaluate_model(args.model, n_episodes=10, stage=args.stage)
    else:
        train_drone_recovery(
            total_timesteps=args.timesteps,
            stage=args.stage,
            model_path=args.model,
            debug=args.debug
        )