from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os
from datetime import datetime
from drone_fast_recovery_env import DroneFastRecoveryEnv

class RecoveryMetricsCallback(BaseCallback):
    """
    Custom callback for tracking recovery-specific metrics
    """
    def __init__(self, verbose=0):
        super(RecoveryMetricsCallback, self).__init__(verbose)
        self.recoveries = []
        self.recovery_times = []
        self.crashes = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get info from environment
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info.keys():
                self.episode_count += 1
                
            if 'episode_stats' in info:
                stats = info['episode_stats']
                
                if stats['successful_recoveries'] > 0:
                    self.recoveries.append(stats['successful_recoveries'])
                    if stats['recovery_time'] > 0:
                        self.recovery_times.append(stats['recovery_time'])
                
                if stats['crashes'] > 0:
                    self.crashes.append(1)
        
        return True
    
    def _on_training_end(self) -> None:
        if self.recovery_times:
            print(f"\n📊 Training Statistics:")
            print(f"   Episodes completed: {self.episode_count}")
            print(f"   Total recoveries: {len(self.recoveries)}")
            print(f"   Average recovery time: {np.mean(self.recovery_times):.1f} steps")
            print(f"   Fastest recovery: {np.min(self.recovery_times):.1f} steps")
            print(f"   Total crashes: {len(self.crashes)}")
            print(f"   Success rate: {len(self.recoveries)/(len(self.recoveries)+len(self.crashes))*100:.1f}%")

class DetailedProgressCallback(BaseCallback):
    """
    Print detailed progress during training
    """
    def __init__(self, check_freq=100):
        super(DetailedProgressCallback, self).__init__()
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                print(f"Steps: {self.num_timesteps:,} | Episodes: {len(self.model.ep_info_buffer)} | "
                      f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.0f}")
            except:
                pass
        return True

def make_env():
    """
    Utility function to create environment
    """
    def _init():
        env = DroneFastRecoveryEnv()
        env = Monitor(env)
        return env
    return _init

def train_ppo(total_timesteps=1000000, save_dir=None):
    """
    Train NEW PPO model for fast recovery
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./models/fast_recovery_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    
    print(f"\n📁 Save directory: {save_dir}")
    
    # Create vectorized environment
    print("\n📦 Creating environment...")
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Callbacks
    recovery_callback = RecoveryMetricsCallback(verbose=1)
    progress_callback = DetailedProgressCallback(check_freq=100)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_recovery",
        verbose=1
    )
    
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(save_dir, "logs"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Create PPO model
    print("\n🧠 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        device="cuda",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
        )
    )
    
    print("\n🎯 Training Configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Control frequency: 10Hz")
    print(f"   Starting altitude: 25m")
    print(f"   Recovery window: ~23m")
    print(f"   Network: [256, 256, 128]")
    print(f"   Learning rate: 5e-4")
    print("\n🔥 Starting training...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[recovery_callback, progress_callback, eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_recovery_final")
    model.save(final_model_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {final_model_path}.zip")
    print(f"   VecNormalize saved to: {save_dir}/vec_normalize.pkl")
    
    return model, env, save_dir

def continue_training(model_path, vec_norm_path, save_dir, additional_timesteps=100000):
    """
    Continue training existing model
    """
    print(f"\n📖 Loading model from {model_path}...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Load and create environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = True
    env.norm_reward = True
    
    # Set environment to model
    model.set_env(env)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    # Setup callbacks
    recovery_callback = RecoveryMetricsCallback(verbose=1)
    progress_callback = DetailedProgressCallback(check_freq=100)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_continued",
        verbose=1
    )
    
    print(f"\n🚀 Continuing training for {additional_timesteps:,} more steps...")
    print(f"   Save directory: {save_dir}")
    
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[recovery_callback, progress_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False  # Keep original timestep count
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    
    # Save continued model
    final_path = os.path.join(save_dir, "ppo_recovery_continued")
    model.save(final_path)
    env.save(os.path.join(save_dir, "vec_normalize_continued.pkl"))
    
    print(f"\n✅ Continued training complete!")
    print(f"   Model saved to: {final_path}.zip")
    print(f"   VecNormalize saved to: {save_dir}/vec_normalize_continued.pkl")
    
    return model, env

def test_model(model_path, vec_norm_path, num_episodes=10):
    """
    Test existing trained model
    """
    print(f"\n🧪 Testing model: {model_path}")
    print(f"   Episodes: {num_episodes}\n")
    
    # Load model and environment
    model = PPO.load(model_path)
    
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    
    results = {
        'successful_recoveries': 0,
        'crashes': 0,
        'recovery_times': [],
        'altitude_losses': [],
        'max_tilts': [],
        'episodes_completed': 0
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        while not done and step < 300:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1
            
            # Print recovery progress
            if len(info) > 0 and 'is_recovering' in info[0]:
                if info[0]['is_recovering'] and step % 10 == 0:
                    print(f"  Step {step}: Alt={info[0]['current_altitude']:.1f}m, "
                          f"Ground_dist={info[0]['ground_distance']:.1f}m")
        
        # Collect results
        if len(info) > 0 and 'episode_stats' in info[0]:
            stats = info[0]['episode_stats']
            results['episodes_completed'] += 1
            
            if stats['successful_recoveries'] > 0:
                results['successful_recoveries'] += 1
                results['recovery_times'].append(stats['recovery_time'])
                print(f"✅ RECOVERED in {stats['recovery_time']} steps "
                      f"({stats['recovery_time']*0.1:.1f}s)")
            
            if stats['crashes'] > 0:
                results['crashes'] += 1
                print(f"💥 CRASHED")
            
            results['altitude_losses'].append(stats['altitude_loss'])
            results['max_tilts'].append(np.degrees(stats['max_tilt']))
            
            print(f"\nEpisode Summary:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Altitude loss: {stats['altitude_loss']:.1f}m")
            print(f"  Max tilt: {np.degrees(stats['max_tilt']):.1f}°")
            print(f"  Closest to ground: {stats['closest_to_ground']:.1f}m")
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"📊 FINAL TEST RESULTS ({results['episodes_completed']} episodes)")
    print(f"{'='*70}")
    
    if results['episodes_completed'] > 0:
        success_rate = (results['successful_recoveries'] / results['episodes_completed']) * 100
        print(f"✅ Successful recoveries: {results['successful_recoveries']}/{results['episodes_completed']} "
              f"({success_rate:.1f}%)")
        print(f"💥 Crashes: {results['crashes']}/{results['episodes_completed']}")
        
        if results['recovery_times']:
            print(f"\n⏱️  Recovery Performance:")
            print(f"   Average time: {np.mean(results['recovery_times']):.1f} steps "
                  f"({np.mean(results['recovery_times'])*0.1:.2f}s)")
            print(f"   Fastest: {np.min(results['recovery_times'])} steps "
                  f"({np.min(results['recovery_times'])*0.1:.2f}s)")
            print(f"   Slowest: {np.max(results['recovery_times'])} steps "
                  f"({np.max(results['recovery_times'])*0.1:.2f}s)")
        
        if results['altitude_losses']:
            print(f"\n📉 Altitude Performance:")
            print(f"   Average loss: {np.mean(results['altitude_losses']):.1f}m")
            print(f"   Max loss: {np.max(results['altitude_losses']):.1f}m")
            print(f"   Min loss: {np.min(results['altitude_losses']):.1f}m")
        
        if results['max_tilts']:
            print(f"\n🌀 Tilt Performance:")
            print(f"   Average max tilt: {np.mean(results['max_tilts']):.1f}°")
            print(f"   Extreme tilt: {np.max(results['max_tilts']):.1f}°")
    
    env.close()
    return results

def main():
    """
    Main menu for training, testing, and continuing training
    """
    print("\n" + "="*70)
    print("DRONE FAST RECOVERY - PPO TRAINING")
    print("="*70)
    print("\nOptions:")
    print("  1. Train new model")
    print("  2. Test existing model")
    print("  3. Continue training existing model")
    print("="*70)
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        # ===== TRAIN NEW MODEL =====
        print("\n" + "="*70)
        print("TRAIN NEW MODEL")
        print("="*70)
        
        timesteps = input("Training timesteps (default 1000000): ").strip()
        timesteps = int(timesteps) if timesteps else 1000000
        
        save_dir = input("Save directory (default: auto-generated): ").strip()
        save_dir = save_dir if save_dir else None
        
        print(f"\n🚀 Starting NEW training with {timesteps:,} timesteps...")
        print("⚠️  This will take several hours. Progress saved automatically.")
        print("   Press Ctrl+C to stop training early (progress will be saved)")
        
        input("\nPress ENTER to begin training...")
        
        model, env, save_dir = train_ppo(total_timesteps=timesteps, save_dir=save_dir)
        
        env.close()
        
        print(f"\n✅ Training complete! All files saved to: {save_dir}")
        print("\nTo test this model, run:")
        print(f"  Option 2 with:")
        print(f"    Model path: {save_dir}/ppo_recovery_final")
        print(f"    VecNormalize: {save_dir}/vec_normalize.pkl")
        
    elif choice == "2":
        # ===== TEST MODEL =====
        print("\n" + "="*70)
        print("TEST EXISTING MODEL")
        print("="*70)
        
        model_path = input("\nModel path (without .zip extension): ").strip()
        
        # Auto-find vec_normalize if in same directory
        model_dir = os.path.dirname(model_path)
        default_vec_norm = os.path.join(model_dir, "vec_normalize.pkl")
        
        if os.path.exists(default_vec_norm):
            print(f"Found VecNormalize: {default_vec_norm}")
            use_default = input("Use this? (Y/n): ").strip().lower()
            if use_default != 'n':
                vec_norm_path = default_vec_norm
            else:
                vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        else:
            vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        
        episodes = input("Test episodes (default 10): ").strip()
        episodes = int(episodes) if episodes else 10
        
        print("\n🧪 Starting testing...")
        
        test_model(model_path, vec_norm_path, num_episodes=episodes)
        
    elif choice == "3":
        # ===== CONTINUE TRAINING =====
        print("\n" + "="*70)
        print("CONTINUE TRAINING EXISTING MODEL")
        print("="*70)
        
        model_path = input("\nModel path to continue (without .zip): ").strip()
        
        # Auto-find vec_normalize
        model_dir = os.path.dirname(model_path)
        default_vec_norm = os.path.join(model_dir, "vec_normalize.pkl")
        
        if os.path.exists(default_vec_norm):
            print(f"Found VecNormalize: {default_vec_norm}")
            use_default = input("Use this? (Y/n): ").strip().lower()
            if use_default != 'n':
                vec_norm_path = default_vec_norm
            else:
                vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        else:
            vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_save = f"./models/continued_{timestamp}"
        save_dir = input(f"Save directory (default: {default_save}): ").strip()
        save_dir = save_dir if save_dir else default_save
        
        timesteps = input("Additional timesteps (default 100000): ").strip()
        timesteps = int(timesteps) if timesteps else 100000
        
        print(f"\n🚀 Continuing training for {timesteps:,} more steps...")
        print("⚠️  Progress will be saved automatically.")
        
        input("\nPress ENTER to continue training...")
        
        model, env = continue_training(model_path, vec_norm_path, save_dir, timesteps)
        
        env.close()
        
        print(f"\n✅ Continued training complete! Files saved to: {save_dir}")
        
    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()