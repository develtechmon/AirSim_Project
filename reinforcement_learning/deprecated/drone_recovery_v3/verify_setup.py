"""
Pre-Training Verification Script
================================
Checks all requirements before starting training.
Catches common issues early to save debugging time.
"""

import sys
import time
from pathlib import Path


def check_imports():
    """Verify all required packages are installed."""
    print("\n🔍 Checking Python Packages...")
    
    required_packages = {
        'airsim': 'airsim',
        'numpy': 'numpy',
        'gymnasium': 'gymnasium',
        'torch': 'torch',
        'stable_baselines3': 'stable-baselines3',
        'tensorboard': 'tensorboard',
        'pandas': 'pandas'
    }
    
    missing = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (install with: pip install {install_name})")
            missing.append(install_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    print("   ✅ All packages installed")
    return True


def check_airsim_connection():
    """Test connection to AirSim."""
    print("\n🔍 Checking AirSim Connection...")
    
    try:
        import airsim
        
        client = airsim.MultirotorClient()
        client.confirmConnection()
        
        print("   ✅ Connected to AirSim")
        
        # Check API control
        client.enableApiControl(True)
        print("   ✅ API control enabled")
        
        # Check vehicle state
        state = client.getMultirotorState()
        print(f"   ✅ Vehicle state retrieved")
        print(f"      Position: ({state.kinematics_estimated.position.x_val:.2f}, "
              f"{state.kinematics_estimated.position.y_val:.2f}, "
              f"{state.kinematics_estimated.position.z_val:.2f})")
        
        # Cleanup
        client.enableApiControl(False)
        
        return True
    
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n   Troubleshooting:")
        print("   1. Ensure AirSim is running (Unreal/Unity environment)")
        print("   2. Check settings.json exists in ~/Documents/AirSim/")
        print("   3. Verify SimMode is 'Multirotor'")
        print("   4. Check firewall allows port 41451")
        return False


def check_airsim_physics():
    """Test AirSim physics and control."""
    print("\n🔍 Testing AirSim Physics...")
    
    try:
        import airsim
        import numpy as np
        
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        print("   📡 Testing takeoff...")
        client.takeoffAsync().join()
        time.sleep(1.0)
        
        print("   📡 Testing angle rate control...")
        client.moveByAngleRatesThrottleAsync(0.1, 0.1, 0.0, 0.6, duration=0.5).join()
        time.sleep(0.5)
        
        # Get state after control
        state = client.getMultirotorState()
        ang_vel = state.kinematics_estimated.angular_velocity
        
        print(f"   ✅ Control working")
        print(f"      Angular velocity: ({ang_vel.x_val:.2f}, {ang_vel.y_val:.2f}, {ang_vel.z_val:.2f}) rad/s")
        
        # Test wind API
        print("   📡 Testing wind API...")
        wind = airsim.Vector3r(5, 0, 0)
        client.simSetWind(wind)
        print("   ✅ Wind API working")
        
        # Test kinematics API
        print("   📡 Testing kinematics API...")
        kinematics = client.simGetGroundTruthKinematics()
        client.simSetKinematics(kinematics, ignore_collision=True)
        print("   ✅ Kinematics API working")
        
        # Cleanup
        client.simSetWind(airsim.Vector3r(0, 0, 0))
        client.reset()
        
        return True
    
    except Exception as e:
        print(f"   ❌ Physics test failed: {e}")
        print("\n   Troubleshooting:")
        print("   1. Ensure SimpleFlight controller is enabled in settings.json")
        print("   2. Check drone is not in collision at start")
        print("   3. Verify physics settings in AirSim")
        return False


def check_environment():
    """Test custom environment."""
    print("\n🔍 Testing Custom Environment...")
    
    try:
        from airsim_recovery_env import AirSimDroneRecoveryEnv
        
        print("   📡 Creating environment...")
        env = AirSimDroneRecoveryEnv(stage=1, debug=False)
        
        print("   📡 Testing reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset successful, observation shape: {obs.shape}")
        
        print("   📡 Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✅ Step successful")
        print(f"      Reward: {reward:.2f}")
        print(f"      Observation valid: {not np.any(np.isnan(obs))}")
        
        env.close()
        
        return True
    
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ppo():
    """Test PPO model creation."""
    print("\n🔍 Testing PPO Model...")
    
    try:
        from stable_baselines3 import PPO
        from airsim_recovery_env import AirSimDroneRecoveryEnv
        import torch
        
        print("   📡 Creating environment...")
        env = AirSimDroneRecoveryEnv(stage=1, debug=False)
        
        print("   📡 Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,  # Small for testing
            batch_size=32,
            verbose=0
        )
        
        print(f"   ✅ PPO model created")
        print(f"      Device: {model.device}")
        print(f"      Policy architecture: {model.policy}")
        
        print("   📡 Testing single training step...")
        model.learn(total_timesteps=64, progress_bar=False)
        print("   ✅ Training step successful")
        
        env.close()
        
        return True
    
    except Exception as e:
        print(f"   ❌ PPO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_disk_space():
    """Check available disk space."""
    print("\n🔍 Checking Disk Space...")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        
        print(f"   Free space: {free_gb} GB")
        
        if free_gb < 5:
            print(f"   ⚠️  Warning: Low disk space (<5 GB)")
            print("      Training logs and models may require 1-5 GB")
            return False
        else:
            print(f"   ✅ Sufficient disk space")
            return True
    
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True


def check_cuda():
    """Check CUDA availability."""
    print("\n🔍 Checking CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available")
            print(f"      Device: {device_name}")
            print(f"      Training will use GPU (much faster!)")
            return True
        else:
            print(f"   ⚠️  CUDA not available")
            print("      Training will use CPU (slower)")
            print("      Consider using GPU for faster training")
            return True
    
    except Exception as e:
        print(f"   ⚠️  Could not check CUDA: {e}")
        return True


def create_directories():
    """Create required directories."""
    print("\n🔍 Creating Directories...")
    
    dirs = ["logs", "models", "models/checkpoints"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True


def run_verification():
    """Run all verification checks."""
    print("=" * 80)
    print("🔧 PRE-TRAINING VERIFICATION")
    print("=" * 80)
    
    checks = [
        ("Python Packages", check_imports),
        ("Directories", create_directories),
        ("Disk Space", check_disk_space),
        ("CUDA", check_cuda),
        ("AirSim Connection", check_airsim_connection),
        ("AirSim Physics", check_airsim_physics),
        ("Custom Environment", check_environment),
        ("PPO Model", check_ppo),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Verification interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_critical_passed = all([
        results.get("Python Packages", False),
        results.get("AirSim Connection", False),
        results.get("AirSim Physics", False),
        results.get("Custom Environment", False),
        results.get("PPO Model", False),
    ])
    
    print("\n" + "=" * 80)
    
    if all_critical_passed:
        print("✅ ALL CRITICAL CHECKS PASSED")
        print("\n🚀 Ready to start training!")
        print("\nRun:")
        print("   python train_drone_recovery.py --stage 1 --timesteps 200000")
        print("\nMonitor:")
        print("   python monitor_training.py")
        print("   tensorboard --logdir=./logs")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        print("See README_TRAINING.md for troubleshooting help.")
        return 1


if __name__ == "__main__":
    import numpy as np
    sys.exit(run_verification())