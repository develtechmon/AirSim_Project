import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from drone_recovery_env import DroneRecoveryEnv

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = DroneRecoveryEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

def train_recovery_model():
    """
    Train PPO model for drone recovery
    """
    print("="*60)
    print("🚁 DRONE RECOVERY PPO TRAINING 🚁")
    print("="*60)
    
    # Create directories
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    
    # Create environment
    print("\n📦 Creating environment...")
    env = DroneRecoveryEnv()
    env = Monitor(env, "./logs/")
    
    # For parallel training (optional - requires multiple AirSim instances)
    # env = SubprocVecEnv([make_env(i) for i in range(4)])
    
    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_drone_recovery"
    )
    
    # Create PPO model
    print("\n🧠 Creating PPO model...")
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cuda"  # Use "cpu" if no GPU
    )
    
    print("\n🎯 Starting training...")
    print(f"   Total timesteps: 500,000")
    print(f"   Learning rate: 3e-4")
    print(f"   Batch size: 64")
    print(f"   Policy: MLP (Multi-Layer Perceptron)")
    print("\n")
    
    # Train the model
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/ppo_drone_recovery_final")
    print("\n✅ Training complete! Model saved to ./models/")
    
    return model

def test_trained_model(model_path="./models/ppo_drone_recovery_final.zip"):
    """
    Test the trained model
    """
    print("\n🧪 Testing trained model...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = DroneRecoveryEnv()
    
    # Run test episodes
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            if step % 20 == 0:
                print(f"  Step {step}: Reward = {reward:.2f}")
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Stats: {info['episode_stats']}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--model", type=str, default="./models/ppo_drone_recovery_final.zip",
                       help="Path to model for testing")
    
    args = parser.parse_args()
    
    if args.train:
        train_recovery_model()
    elif args.test:
        test_trained_model(args.model)
    else:
        # Default: train then test
        model = train_recovery_model()
        input("\nPress ENTER to test the trained model...")
        test_trained_model()