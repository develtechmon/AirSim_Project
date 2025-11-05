"""
FAST TRAINING MODE - GPU Optimized
FIXED: VecNormalize mismatch + AirSim crash prevention
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from datetime import datetime

from drone_chase_env import DroneChaseEnv


def check_gpu():
    """Check GPU availability and print detailed info"""
    print("\n" + "="*70)
    print("🖥️  GPU CONFIGURATION CHECK")
    print("="*70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED!")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"  Total memory: {memory_total:.2f} GB")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print(f"  Free: {memory_total - memory_reserved:.2f} GB")
        
        # Test GPU computation
        print(f"\n🧪 Testing GPU computation speed...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            import time
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"  ✅ GPU test passed! Time: {elapsed:.2f}s")
            print(f"  🚀 GPU is READY for training!")
        except Exception as e:
            print(f"  ❌ GPU test failed: {e}")
        
        print("="*70 + "\n")
        return True
    else:
        print(f"❌ NO GPU DETECTED!")
        print(f"\n⚠️  Training will use CPU (VERY SLOW!)")
        print(f"\nTo enable GPU, install PyTorch with CUDA:")
        print(f"  pip uninstall torch torchvision torchaudio")
        print(f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("="*70 + "\n")
        return False


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = DroneChaseEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_single_env():
    """
    SINGLE ENVIRONMENT TRAINING - Prevents AirSim crashes
    Slower but more stable for AirSim
    """
    print("="*70)
    print("🚀 SINGLE ENVIRONMENT MODE - STABLE FOR AIRSIM")
    print("="*70)
    
    # Check GPU
    gpu_available = check_gpu()
    
    print("\nConfiguration:")
    print("  ✓ 1 environment (stable, no crashes)")
    print("  ✓ Optimized hyperparameters")
    print("  ✓ Observation normalization")
    print("  ✓ Action space: ±15 m/s (aggressive)")
    if gpu_available:
        print("  ✓ GPU acceleration")
    print()
    
    # Directories
    models_dir = "models/single"
    logs_dir = "logs/single"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create SINGLE environment (no parallel = no crashes)
    print("[1/5] Creating single environment...")
    env = DummyVecEnv([make_env(0)])
    
    # Normalize observations
    print("[2/5] Adding observation normalization...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create PPO
    print("[3/5] Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,  # Increased for single env
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dir,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            )
        )
    )
    
    print(f"✓ Model created on device: {model.device}")
    
    if gpu_available:
        print(f"\n🚀 Verifying GPU usage...")
        print(f"  Model device: {next(model.policy.parameters()).device}")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Setup callbacks
    print("[4/5] Setting up callbacks...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ppo_single_{timestamp}"
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{models_dir}/checkpoints/{model_name}",
        name_prefix="checkpoint",
        save_vecnormalize=True
    )
    
    # FIXED: Evaluation environment with matching VecNormalize
    eval_env = DummyVecEnv([make_env(999)])  # Different seed for eval
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        gamma=0.99,
        training=False  # Important! Don't update stats during eval
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best/{model_name}",
        log_path=f"{logs_dir}/eval",
        eval_freq=5000,
        deterministic=True,
        n_eval_episodes=3
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train!
    print("[5/5] Starting training...")
    print(f"  - Total timesteps: 150,000")
    if gpu_available:
        print(f"  - Expected time with GPU: ~30-45 min")
    else:
        print(f"  - Expected time with CPU: ~2-3 hours")
    print(f"  - Single env = MORE STABLE (no AirSim crashes)")
    print()
    
    try:
        model.learn(
            total_timesteps=150000,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{models_dir}/{model_name}_final"
        model.save(final_path)
        env.save(f"{final_path}_vecnormalize.pkl")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Final model saved: {final_path}")
        print(f"Normalization stats saved: {final_path}_vecnormalize.pkl")
        
        if gpu_available:
            print(f"\n📊 GPU Usage Statistics:")
            print(f"  Peak memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Current memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        save_path = f"{models_dir}/{model_name}_interrupted"
        model.save(save_path)
        env.save(f"{save_path}_vecnormalize.pkl")
        print(f"Model saved at interruption: {save_path}")
    
    finally:
        env.close()
        eval_env.close()
        
        if gpu_available:
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")


def train_fast():
    """
    PARALLEL ENVIRONMENTS - Faster but may crash AirSim
    Use only if your AirSim is stable with multiple drones
    """
    print("="*70)
    print("🚀 FAST TRAINING MODE - 2 PARALLEL ENVIRONMENTS")
    print("="*70)
    print("⚠️  WARNING: May cause AirSim to crash/freeze")
    print("If unstable, use Single Environment mode instead\n")
    
    # Check GPU
    gpu_available = check_gpu()
    
    if not gpu_available:
        response = input("⚠️  No GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    print("\nOptimizations:")
    print("  ✓ 2 parallel environments (2x faster)")  # REDUCED from 4 to 2!
    print("  ✓ Optimized hyperparameters")
    print("  ✓ Observation normalization")
    if gpu_available:
        print("  ✓ GPU acceleration")
    print()
    
    # Directories
    models_dir = "models/fast"
    logs_dir = "logs/fast"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create 2 parallel environments (REDUCED to prevent crashes)
    print("[1/5] Creating 2 parallel environments...")
    num_envs = 2  # REDUCED from 4!
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Normalize observations
    print("[2/5] Adding observation normalization...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create PPO
    print("[3/5] Creating optimized PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dir,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            )
        )
    )
    
    print(f"✓ Model created on device: {model.device}")
    print(f"✓ Using {num_envs} parallel environments")
    
    if gpu_available:
        print(f"\n🚀 Verifying GPU usage...")
        print(f"  Model device: {next(model.policy.parameters()).device}")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Setup callbacks
    print("[4/5] Setting up callbacks...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ppo_fast_{timestamp}"
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // num_envs,
        save_path=f"{models_dir}/checkpoints/{model_name}",
        name_prefix="checkpoint",
        save_vecnormalize=True
    )
    
    # FIXED: Evaluation environment with matching VecNormalize
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
        best_model_save_path=f"{models_dir}/best/{model_name}",
        log_path=f"{logs_dir}/eval",
        eval_freq=5000 // num_envs,
        deterministic=True,
        n_eval_episodes=3
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train!
    print("[5/5] Starting FAST training...")
    print(f"  - Total timesteps: 150,000")
    print(f"  - With {num_envs} parallel envs = {150000//num_envs} steps per env")
    if gpu_available:
        print(f"  - Expected time with GPU: ~20-30 min")
    else:
        print(f"  - Expected time with CPU: ~1-2 hours")
    print()
    
    try:
        model.learn(
            total_timesteps=150000,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{models_dir}/{model_name}_final"
        model.save(final_path)
        env.save(f"{final_path}_vecnormalize.pkl")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Final model saved: {final_path}")
        print(f"Normalization stats saved: {final_path}_vecnormalize.pkl")
        
        if gpu_available:
            print(f"\n📊 GPU Usage Statistics:")
            print(f"  Peak memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Current memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        save_path = f"{models_dir}/{model_name}_interrupted"
        model.save(save_path)
        env.save(f"{save_path}_vecnormalize.pkl")
        print(f"Model saved at interruption: {save_path}")
    
    finally:
        env.close()
        eval_env.close()
        
        if gpu_available:
            torch.cuda.empty_cache()
            print("✓ GPU memory cleared")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         TRAINING OPTIONS - AIRSIM CRASH FIX                        ║
╚═══════════════════════════════════════════════════════════════════╝

Your issue:
  - AirSim crashes/freezes during training 💥
  - Multiple parallel environments cause instability ⚠️

Solutions:
1. SINGLE ENVIRONMENT (Recommended) - Most stable, no crashes
2. FAST MODE (2 parallel) - 2x faster, may be unstable
3. Exit

Choose wisely!
    """)
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("\n🚀 Starting SINGLE environment training...")
        print("This is the MOST STABLE option for AirSim.\n")
        input("Press Enter to continue...")
        train_single_env()
    elif choice == "2":
        print("\n⚡ Starting FAST mode with 2 parallel environments...")
        print("⚠️  If AirSim crashes, use Single Environment mode instead.\n")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            train_fast()
        else:
            print("Training cancelled.")
    else:
        print("Invalid choice!")