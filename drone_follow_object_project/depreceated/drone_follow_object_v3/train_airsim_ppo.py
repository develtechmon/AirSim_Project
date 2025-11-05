"""
PPO Training Script for Drone Sphere Chasing
Interactive menu for training, testing, and continuing training
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from datetime import datetime

from drone_chase_env import DroneChaseEnv


class TrainingConfig:
    """Configuration for PPO training"""
    def __init__(self):
        self.total_timesteps = 300000
        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01  # Entropy coefficient for exploration
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Environment config
        self.max_altitude = 50
        self.min_altitude = -50
        self.hit_threshold = 2.0
        self.max_distance = 100
        self.spawn_radius = 30


class DroneTrainer:
    """
    Main trainer class - Think of this as the training facility manager
    It handles all the logistics: creating gyms, hiring coaches, keeping records
    """
    
    def __init__(self):
        self.models_dir = "models"
        self.logs_dir = "logs"
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(f"{self.logs_dir}/eval", exist_ok=True)
        print(f"✓ Directories ready: {self.models_dir}, {self.logs_dir}")
    
    def create_env(self, config):
        """Create and wrap the environment"""
        def _init():
            env = DroneChaseEnv(
                max_altitude=config.max_altitude,
                min_altitude=config.min_altitude,
                hit_threshold=config.hit_threshold,
                max_distance=config.max_distance,
                spawn_radius=config.spawn_radius
            )
            env = Monitor(env, filename=f"{self.logs_dir}/monitor")
            return env
        
        # Vectorized environment for better performance
        env = DummyVecEnv([_init])
        return env
    
    def train_new_model(self, config):
        """Train a brand new model from scratch"""
        print("\n" + "="*60)
        print("🚀 TRAINING NEW MODEL")
        print("="*60)
        
        # Create environment
        print("\n[1/4] Creating environment...")
        env = self.create_env(config)
        
        # Create model
        print("[2/4] Initializing PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=1,
            tensorboard_log=self.logs_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✓ Model created on device: {model.device}")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
        
        # Setup callbacks
        print("[3/4] Setting up training callbacks...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_drone_chase_{timestamp}"
        
        # Checkpoint callback - saves model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.models_dir}/checkpoints/{model_name}",
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        
        # Evaluation callback - evaluates model during training
        eval_env = self.create_env(config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/best/{model_name}",
            log_path=f"{self.logs_dir}/eval",
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        callbacks = [checkpoint_callback, eval_callback]
        
        # Train!
        print("[4/4] Starting training...")
        print(f"  - Total timesteps: {config.total_timesteps:,}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Checkpoints every: 10,000 steps")
        print(f"  - Evaluation every: 5,000 steps")
        print("\n" + "-"*60)
        
        try:
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            final_path = f"{self.models_dir}/{model_name}_final"
            model.save(final_path)
            
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            print(f"Final model saved: {final_path}")
            print(f"Best model saved: {self.models_dir}/best/{model_name}")
            print(f"Checkpoints saved: {self.models_dir}/checkpoints/{model_name}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            save_path = f"{self.models_dir}/{model_name}_interrupted"
            model.save(save_path)
            print(f"Model saved at interruption: {save_path}")
        
        finally:
            env.close()
            eval_env.close()
    
    def continue_training(self, model_path, additional_timesteps):
        """Continue training an existing model"""
        print("\n" + "="*60)
        print("🔄 CONTINUING TRAINING")
        print("="*60)
        
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model not found: {model_path}")
            return
        
        # Load existing model
        print(f"\n[1/4] Loading model from: {model_path}")
        config = TrainingConfig()  # Use same config
        env = self.create_env(config)
        
        model = PPO.load(
            model_path,
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"✓ Model loaded on device: {model.device}")
        
        # Setup callbacks
        print("[2/4] Setting up training callbacks...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        continued_name = f"{os.path.basename(model_path)}_continued_{timestamp}"
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{self.models_dir}/checkpoints/{continued_name}",
            name_prefix="checkpoint"
        )
        
        eval_env = self.create_env(config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/best/{continued_name}",
            log_path=f"{self.logs_dir}/eval",
            eval_freq=5000,
            deterministic=True,
            n_eval_episodes=5
        )
        
        callbacks = [checkpoint_callback, eval_callback]
        
        # Continue training
        print(f"[3/4] Resuming training for {additional_timesteps:,} more steps...")
        print("-"*60)
        
        try:
            model.learn(
                total_timesteps=additional_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False  # Keep counting from where we left off
            )
            
            # Save continued model
            final_path = f"{self.models_dir}/{continued_name}_final"
            model.save(final_path)
            
            print("\n" + "="*60)
            print("✅ CONTINUED TRAINING COMPLETE!")
            print("="*60)
            print(f"Model saved: {final_path}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            save_path = f"{self.models_dir}/{continued_name}_interrupted"
            model.save(save_path)
            print(f"Model saved at interruption: {save_path}")
        
        finally:
            env.close()
            eval_env.close()
    
    def test_model(self, model_path, n_episodes=5):
        """Test an existing model"""
        print("\n" + "="*60)
        print("🎮 TESTING MODEL")
        print("="*60)
        
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model not found: {model_path}")
            return
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        config = TrainingConfig()
        env = self.create_env(config)
        
        model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Model loaded on device: {model.device}")
        
        # Test episodes
        print(f"\nRunning {n_episodes} test episodes...")
        print("-"*60)
        
        episode_rewards = []
        episode_lengths = []
        total_hits = 0
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            hits = 0
            done = False
            
            print(f"\n📊 Episode {episode + 1}/{n_episodes}")
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if info[0].get('hit_target', False):
                    hits += 1
                    print(f"  🎯 HIT! (Total hits: {hits})")
                
                done = done[0]
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_hits += hits
            
            print(f"  ✓ Reward: {episode_reward:.2f}")
            print(f"  ✓ Length: {episode_length} steps")
            print(f"  ✓ Hits: {hits}")
        
        # Print summary
        print("\n" + "="*60)
        print("📈 TEST SUMMARY")
        print("="*60)
        print(f"Total episodes: {n_episodes}")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Total hits: {total_hits}")
        print(f"Average hits per episode: {total_hits/n_episodes:.2f}")
        print(f"Best episode reward: {max(episode_rewards):.2f}")
        print(f"Worst episode reward: {min(episode_rewards):.2f}")
        
        env.close()
    
    def list_models(self):
        """List all available models"""
        print("\n" + "="*60)
        print("📁 AVAILABLE MODELS")
        print("="*60)
        
        if not os.path.exists(self.models_dir):
            print("No models found.")
            return []
        
        models = []
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.zip'):
                    model_path = os.path.join(root, file[:-4])  # Remove .zip
                    rel_path = os.path.relpath(model_path, self.models_dir)
                    models.append(model_path)
                    
                    # Get file size and modified time
                    full_path = model_path + ".zip"
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
                    
                    print(f"\n{len(models)}. {rel_path}")
                    print(f"   Size: {size_mb:.2f} MB")
                    print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not models:
            print("No models found in models directory.")
        
        return models


def get_training_config():
    """Interactive configuration setup"""
    print("\n" + "="*60)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*60)
    
    config = TrainingConfig()
    
    # Total timesteps
    print(f"\n📊 Training timesteps (default {config.total_timesteps:,})")
    timesteps_input = input("   Enter timesteps (or press Enter for default): ").strip()
    if timesteps_input:
        try:
            config.total_timesteps = int(timesteps_input)
        except ValueError:
            print("   ⚠️  Invalid input, using default")
    
    # Learning rate
    print(f"\n🎓 Learning rate (default {config.learning_rate})")
    print("   Common values: 3e-4 (default), 1e-4 (conservative), 1e-3 (aggressive)")
    lr_input = input("   Enter learning rate (or press Enter for default): ").strip()
    if lr_input:
        try:
            config.learning_rate = float(lr_input)
        except ValueError:
            print("   ⚠️  Invalid input, using default")
    
    # Show full config
    print("\n✓ Configuration set:")
    print(f"  - Timesteps: {config.total_timesteps:,}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Environment hit threshold: {config.hit_threshold}m")
    print(f"  - Altitude limits: [{config.min_altitude}, {config.max_altitude}]")
    
    return config


def main():
    """Main interactive menu"""
    trainer = DroneTrainer()
    
    print("\n" + "="*60)
    print("🚁 DRONE SPHERE CHASING - PPO TRAINING SYSTEM")
    print("="*60)
    print("Welcome! This system will train your drone to chase and hit targets.")
    print("Using PPO (Proximal Policy Optimization) algorithm.")
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train new model")
        print("2. Test existing model")
        print("3. Continue training existing model")
        print("4. List all models")
        print("5. Exit")
        print("-"*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            # Train new model
            config = get_training_config()
            confirm = input("\n🚀 Start training? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                trainer.train_new_model(config)
            else:
                print("Training cancelled.")
        
        elif choice == "2":
            # Test existing model
            models = trainer.list_models()
            if models:
                try:
                    model_idx = int(input("\nEnter model number to test: ").strip()) - 1
                    if 0 <= model_idx < len(models):
                        n_episodes = input("Number of test episodes (default 5): ").strip()
                        n_episodes = int(n_episodes) if n_episodes else 5
                        trainer.test_model(models[model_idx], n_episodes)
                    else:
                        print("❌ Invalid model number")
                except ValueError:
                    print("❌ Invalid input")
        
        elif choice == "3":
            # Continue training
            models = trainer.list_models()
            if models:
                try:
                    model_idx = int(input("\nEnter model number to continue training: ").strip()) - 1
                    if 0 <= model_idx < len(models):
                        timesteps = input("Additional timesteps (default 100000): ").strip()
                        timesteps = int(timesteps) if timesteps else 100000
                        confirm = input(f"\n🔄 Continue training for {timesteps:,} steps? (yes/no): ").strip().lower()
                        if confirm in ['yes', 'y']:
                            trainer.continue_training(models[model_idx], timesteps)
                        else:
                            print("Training cancelled.")
                    else:
                        print("❌ Invalid model number")
                except ValueError:
                    print("❌ Invalid input")
        
        elif choice == "4":
            # List models
            trainer.list_models()
        
        elif choice == "5":
            # Exit
            print("\n👋 Goodbye! Happy training!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()