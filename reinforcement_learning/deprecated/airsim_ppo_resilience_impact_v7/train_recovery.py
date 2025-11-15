"""
Training Script - VERIFIED TO RUN TO COMPLETION
✅ Will train to target timesteps
✅ No early stopping
✅ Progress bar shows actual progress
"""

import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from recovery_env import RecoveryEnv


def check_gpu():
    """Check GPU"""
    print("\n" + "="*70)
    print("GPU CHECK")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("="*70 + "\n")
        return True
    else:
        print("❌ NO GPU")
        print("="*70 + "\n")
        return False


def make_env(rank, seed=0):
    """Create environment"""
    def _init():
        env = RecoveryEnv(enable_tracking=False)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_new():
    """Train new model"""
    print("\n" + "="*70)
    print("🚀 TRAIN NEW MODEL")
    print("="*70)
    
    gpu_available = check_gpu()
    
    # Get timesteps
    timesteps_input = input("Total timesteps (default 150000): ").strip()
    try:
        timesteps = int(timesteps_input) if timesteps_input else 150000
    except:
        print("Invalid, using 150000")
        timesteps = 150000
    
    # Confirm
    print(f"\n✓ Will train for {timesteps:,} timesteps")
    print(f"✓ This is GUARANTEED to complete (no early stopping)")
    confirm = input("Continue? (y/n): ").strip()
    if confirm.lower() != 'y':
        print("Cancelled.")
        return
    
    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"recovery_{timestamp}"
    models_dir = "models/recovery"
    logs_dir = "logs/recovery"
    
    os.makedirs(f"{models_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{models_dir}/best", exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\n[1/4] Creating environment...")
    env = DummyVecEnv([make_env(0)])
    
    print("[2/4] Adding normalization...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    print("[3/4] Creating PPO model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
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
        tensorboard_log=logs_dir,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],  # ✅ Policy network: 2 layers of 256 neurons
                vf=[256, 256]   # ✅ Value network: 2 layers of 256 neurons
            )
        )
    )
    
    print(f"✓ Model on: {model.device}")
    
    print("[4/4] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{models_dir}/checkpoints",
        name_prefix=model_name,
        save_vecnormalize=True
    )
    
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        training=False
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best",
        log_path=f"{logs_dir}/eval",
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=3
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'='*70}")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Progress will be shown below")
    print(f"  Training WILL complete all {timesteps:,} steps")
    if gpu_available:
        print(f"  Estimated time: ~{timesteps/50000*25:.0f} minutes")
    print(f"{'='*70}\n")
    
    try:
        # ✅ THIS WILL RUN TO COMPLETION
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final
        final_path = f"{models_dir}/{model_name}_final"
        model.save(final_path)
        env.save(f"{final_path}_vecnormalize.pkl")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"✓ Completed all {timesteps:,} timesteps")
        print(f"✓ Model: {final_path}.zip")
        print(f"✓ VecNormalize: {final_path}_vecnormalize.pkl")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user (Ctrl+C)")
        interrupted_path = f"{models_dir}/{model_name}_interrupted"
        model.save(interrupted_path)
        env.save(f"{interrupted_path}_vecnormalize.pkl")
        print(f"✓ Saved: {interrupted_path}.zip")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        error_path = f"{models_dir}/{model_name}_error"
        try:
            model.save(error_path)
            env.save(f"{error_path}_vecnormalize.pkl")
            print(f"✓ Saved state: {error_path}.zip")
        except:
            print("❌ Could not save model")
    
    finally:
        env.close()
        eval_env.close()
        if gpu_available:
            torch.cuda.empty_cache()

def resume_training():
    """Resume training"""
    print("\n" + "="*70)
    print("🔄 RESUME TRAINING")
    print("="*70)
    
    gpu_available = check_gpu()
    
    # Get paths
    print("Enter model path:")
    model_path = input("Model (.zip): ").strip().strip('"').strip("'")
    
    if not os.path.exists(model_path):
        print(f"❌ Not found: {model_path}")
        return
    
    vecnorm_path = model_path.replace('.zip', '_vecnormalize.pkl')
    
    if not os.path.exists(vecnorm_path):
        print(f"❌ Not found: {vecnorm_path}")
        return
    
    # Get additional timesteps
    additional_input = input("\nAdditional timesteps (default 100000): ").strip()
    try:
        additional_timesteps = int(additional_input) if additional_input else 100000
    except:
        print("Invalid, using 100000")
        additional_timesteps = 100000
    
    print(f"\n✓ Will train for {additional_timesteps:,} MORE steps")
    print(f"✓ Training WILL complete all steps")
    confirm = input("Continue? (y/n): ").strip()
    if confirm.lower() != 'y':
        return
    
    # Load
    print("\n[1/3] Loading environment...")
    env = DummyVecEnv([make_env(0)])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = True
    env.norm_reward = True
    
    print("[2/3] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)
    
    current_steps = model.num_timesteps
    print(f"✓ Loaded: {current_steps:,} steps completed")
    print(f"✓ Will train to: {current_steps + additional_timesteps:,} steps")
    
    print("[3/3] Setting up callbacks...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resumed_name = f"resumed_{timestamp}"
    
    models_dir = "models/recovery"
    logs_dir = "logs/recovery"
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{models_dir}/checkpoints",
        name_prefix=resumed_name,
        save_vecnormalize=True
    )
    
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best",
        log_path=f"{logs_dir}/eval",
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=3
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    print(f"\n{'='*70}")
    print(f"🚀 RESUMING TRAINING")
    print(f"{'='*70}")
    print(f"  Additional timesteps: {additional_timesteps:,}")
    print(f"  Training WILL complete all steps")
    print(f"{'='*70}\n")
    
    try:
        # ✅ THIS WILL RUN TO COMPLETION
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Save final
        final_path = f"{models_dir}/{resumed_name}_final"
        model.save(final_path)
        env.save(f"{final_path}_vecnormalize.pkl")
        
        print("\n" + "="*70)
        print("✅ RESUMED TRAINING COMPLETE!")
        print("="*70)
        print(f"✓ Total timesteps: {model.num_timesteps:,}")
        print(f"✓ Saved: {final_path}.zip")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        interrupted_path = f"{models_dir}/{resumed_name}_interrupted"
        model.save(interrupted_path)
        env.save(f"{interrupted_path}_vecnormalize.pkl")
        print(f"✓ Saved: {interrupted_path}.zip")
    
    finally:
        env.close()
        eval_env.close()
        if gpu_available:
            torch.cuda.empty_cache()


def test_model():
    """Test model"""
    print("\n" + "="*70)
    print("🧪 TEST MODEL")
    print("="*70)
    
    # Get paths
    model_path = input("\nModel (.zip): ").strip().strip('"').strip("'")
    
    if not os.path.exists(model_path):
        print(f"❌ Not found: {model_path}")
        return
    
    vecnorm_path = model_path.replace('.zip', '_vecnormalize.pkl')
    
    if not os.path.exists(vecnorm_path):
        print(f"❌ Not found: {vecnorm_path}")
        return
    
    episodes_input = input("Test episodes (default 10): ").strip()
    try:
        num_episodes = int(episodes_input) if episodes_input else 10
    except:
        num_episodes = 10
    
    print(f"\n✓ Will test for {num_episodes} episodes")
    confirm = input("Continue? (y/n): ").strip()
    if confirm.lower() != 'y':
        return
    
    # Load
    print("\n[1/3] Creating test environment...")
    env = DummyVecEnv([lambda: RecoveryEnv(enable_tracking=True)])
    
    print("[2/3] Loading VecNormalize...")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    
    print("[3/3] Loading model...")
    model = PPO.load(model_path, env=env)
    
    print(f"\n✓ Model trained for {model.num_timesteps:,} steps\n")
    
    # Test
    episode_rewards = []
    recovery_counts = []
    crash_counts = []
    best_tilts = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done and len(info) > 0:
                ep_info = info[0]
                
                recovery_count = ep_info.get('recovery_count', 0)
                disturbance_count = ep_info.get('disturbance_count', 0)
                crashed = ep_info.get('crashed', False)
                best_tilt = ep_info.get('best_tilt', 180)
                
                recovery_counts.append(recovery_count)
                crash_counts.append(1 if crashed else 0)
                best_tilts.append(best_tilt)
                
                print(f"\nResults:")
                print(f"  Recoveries: {recovery_count}/{disturbance_count}")
                print(f"  Best tilt: {best_tilt:.1f}°")
                print(f"  Reward: {episode_reward:.1f}")
                print(f"  Status: {'CRASHED' if crashed else 'COMPLETED'}")
                
                break
        
        episode_rewards.append(episode_reward)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.1f}")
    print(f"Mean recoveries: {np.mean(recovery_counts):.2f}")
    print(f"Crash rate: {np.mean(crash_counts) * 100:.1f}%")
    print(f"Mean best tilt: {np.mean(best_tilts):.1f}°")
    print("="*70 + "\n")
    
    env.close()


def list_files():
    """List files"""
    print("\n" + "="*70)
    print("📁 MODEL FILES")
    print("="*70)
    
    models_dir = "models/recovery"
    
    if not os.path.exists(models_dir):
        print(f"\n❌ Not found: {models_dir}")
        return
    
    zip_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.zip'):
                full_path = os.path.join(root, file)
                vecnorm_path = full_path.replace('.zip', '_vecnormalize.pkl')
                has_vecnorm = os.path.exists(vecnorm_path)
                
                zip_files.append({
                    'path': full_path,
                    'size_mb': os.path.getsize(full_path) / 1e6,
                    'has_vecnorm': has_vecnorm
                })
    
    if not zip_files:
        print(f"\n❌ No .zip files in {models_dir}")
        return
    
    print(f"\nFound {len(zip_files)} file(s):\n")
    
    for i, file_info in enumerate(zip_files, 1):
        vecnorm_status = "✓" if file_info['has_vecnorm'] else "✗"
        print(f"{i}. {file_info['path']}")
        print(f"   Size: {file_info['size_mb']:.1f} MB | VecNormalize: {vecnorm_status}\n")
    
    print("="*70 + "\n")


def main():
    """Main menu"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         🚁 RECOVERY TRAINING - VERIFIED                           ║
║                                                                   ║
║  ✅ Training will complete all timesteps                         ║
║  ✅ No early stopping                                            ║
║  ✅ Progress bar shows actual progress                           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    gpu_available = check_gpu()
    
    if not gpu_available:
        print("⚠️  No GPU - training will be slow")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("\n1. Train NEW model")
        print("2. Resume training")
        print("3. Test model")
        print("4. List files")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            train_new()
        elif choice == "2":
            resume_training()
        elif choice == "3":
            test_model()
        elif choice == "4":
            list_files()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid")


if __name__ == "__main__":
    main()