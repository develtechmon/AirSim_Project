import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv
from airsim_drone_balloon_env import AirSimDroneBalloonEnv
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')


def make_airsim_env(render_mode=None):
    """Factory function to create AirSim environment"""
    def _init():
        return AirSimDroneBalloonEnv(render_mode=render_mode)
    return _init


class ProgressCallback(BaseCallback):
    """Show progress during training"""
    def __init__(self, check_freq=100):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_scores = []
    
    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            if 'score' in info:
                self.episode_scores.append(info['score'])
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            avg_score = sum(self.episode_scores[-10:]) / min(10, len(self.episode_scores)) if self.episode_scores else 0
            print(f"Steps: {self.n_calls:,} | Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f}/6")
        
        return True


def train_new_model(timesteps=300000, save_dir="./airsim_models"):
    """Train a new PPO model from scratch"""
    
    print("\n" + "=" * 70)
    print("Starting New Training")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    print("\n⚠️  Make sure AirSim is running!")
    input("Press Enter when ready...")
    
    print("\nCreating training environment...")
    train_env = DummyVecEnv([make_airsim_env(render_mode="human")])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    print("✓ Environment created")
    
    # Callbacks
    progress_callback = ProgressCallback(check_freq=100)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix='airsim_drone',
        save_vecnormalize=True
    )
    
    # Create model - SIMPLE settings that work
    print("\nInitializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
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
        verbose=1,
        tensorboard_log="./airsim_tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    
    print("\n" + "=" * 70)
    print(f"Training for {timesteps:,} timesteps...")
    print("=" * 70)
    print("\n💡 Tips:")
    print("   - Watch the drone in AirSim's window")
    print("   - Monitor: tensorboard --logdir ./airsim_tensorboard_logs/")
    print("   - Stop anytime with Ctrl+C (auto-saves)")
    print()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[progress_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(save_dir, f"airsim_drone_final_{timestamp}")
        model.save(final_path)
        train_env.save(os.path.join(save_dir, f"vec_normalize_{timestamp}.pkl"))
        
        print("\n" + "=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)
        print(f"Model saved: {final_path}.zip")
        print(f"Normalization: {save_dir}/vec_normalize_{timestamp}.pkl")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted!")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_path = os.path.join(save_dir, f"airsim_drone_interrupted_{timestamp}")
        model.save(interrupted_path)
        train_env.save(os.path.join(save_dir, f"vec_normalize_interrupted_{timestamp}.pkl"))
        print(f"✓ Model saved: {interrupted_path}.zip")
    
    finally:
        train_env.close()


def test_model(model_path, vec_norm_path, num_episodes=5):
    """Test a trained model"""
    
    print("\n" + "=" * 70)
    print("Testing Trained Model")
    print("=" * 70)
    
    print("\n⚠️  Make sure AirSim is running!")
    input("Press Enter when ready...")
    
    # Create environment
    print("\nCreating test environment...")
    env = DummyVecEnv([make_airsim_env(render_mode="human")])
    env = VecMonitor(env)
    
    # Load normalization stats
    print(f"Loading normalization from: {vec_norm_path}")
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    
    print(f"\n✓ Model loaded!")
    print(f"\nRunning {num_episodes} test episodes...\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            step += 1
            
            if done[0]:
                score = info[0].get('score', 0)
                distance = info[0].get('distance_to_target', 0)
                print(f"  Steps: {step}")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Final Score: {score}/6")
                print(f"  Final Distance: {distance:.2f}m")
                break
        
        print()
    
    env.close()
    print("\n✓ Testing complete!")


def continue_training(model_path, vec_norm_path, additional_timesteps=100000, save_dir="./airsim_models"):
    """Continue training from a checkpoint"""
    
    print("\n" + "=" * 70)
    print("Continuing Training from Checkpoint")
    print("=" * 70)
    
    print("\n⚠️  Make sure AirSim is running!")
    input("Press Enter when ready...")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([make_airsim_env(render_mode="human")])
    env = VecMonitor(env)
    
    # Load normalization
    print(f"Loading normalization from: {vec_norm_path}")
    env = VecNormalize.load(vec_norm_path, env)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    
    print("✓ Model and environment loaded!")
    
    # Callbacks
    progress_callback = ProgressCallback(check_freq=100)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix='airsim_drone_continued',
        save_vecnormalize=True
    )
    
    print(f"\n🚀 Continuing training for {additional_timesteps:,} more steps...")
    print("Stop anytime with Ctrl+C (auto-saves)\n")
    
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[progress_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False  # Keep counting from previous timesteps
        )
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(save_dir, f"airsim_drone_continued_{timestamp}")
        model.save(final_path)
        env.save(os.path.join(save_dir, f"vec_normalize_continued_{timestamp}.pkl"))
        
        print("\n" + "=" * 70)
        print("✓ Continued Training Complete!")
        print("=" * 70)
        print(f"Model saved: {final_path}.zip")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted!")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_path = os.path.join(save_dir, f"airsim_drone_continued_interrupted_{timestamp}")
        model.save(interrupted_path)
        env.save(os.path.join(save_dir, f"vec_normalize_continued_interrupted_{timestamp}.pkl"))
        print(f"✓ Model saved: {interrupted_path}.zip")
    
    finally:
        env.close()


def main():
    """Main menu interface"""
    
    print("\n" + "=" * 70)
    print("AirSim Drone Training with PPO")
    print("=" * 70)
    
    print("\nOptions:")
    print("  1. Train new model")
    print("  2. Test existing model")
    print("  3. Continue training existing model")
    print("=" * 70)
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        # New training
        print("\n📋 New Training Configuration")
        timesteps = input("  Training timesteps (default 300000): ").strip()
        timesteps = int(timesteps) if timesteps else 300000
        
        save_dir = input("  Save directory (default ./airsim_models): ").strip()
        save_dir = save_dir if save_dir else "./airsim_models"
        
        print(f"\n🚀 Starting training with {timesteps:,} timesteps...")
        
        input("\nPress ENTER to begin...")
        
        train_new_model(timesteps=timesteps, save_dir=save_dir)
        
    elif choice == "2":
        # Test model
        print("\n📋 Test Model Configuration")
        model_path = input("  Model path (without .zip): ").strip()
        
        if not model_path:
            print("❌ Model path required!")
            return
        
        vec_norm_path = input("  VecNormalize path (.pkl): ").strip()
        
        if not vec_norm_path:
            print("❌ VecNormalize path required!")
            return
        
        episodes = input("  Test episodes (default 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        test_model(model_path, vec_norm_path, num_episodes=episodes)
        
    elif choice == "3":
        # Continue training
        print("\n📋 Continue Training Configuration")
        model_path = input("  Model path to continue (without .zip): ").strip()
        
        if not model_path:
            print("❌ Model path required!")
            return
        
        vec_norm_path = input("  VecNormalize path (.pkl): ").strip()
        
        if not vec_norm_path:
            print("❌ VecNormalize path required!")
            return
        
        timesteps = input("  Additional timesteps (default 100000): ").strip()
        timesteps = int(timesteps) if timesteps else 100000
        
        save_dir = input("  Save directory (default ./airsim_models): ").strip()
        save_dir = save_dir if save_dir else "./airsim_models"
        
        continue_training(
            model_path=model_path,
            vec_norm_path=vec_norm_path,
            additional_timesteps=timesteps,
            save_dir=save_dir
        )
    
    else:
        print("❌ Invalid choice! Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    main()