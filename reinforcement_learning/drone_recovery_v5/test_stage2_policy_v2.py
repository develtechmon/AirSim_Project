"""
TEST STAGE 2: DISTURBANCE RECOVERY POLICY (13 OBSERVATIONS)
============================================================
Tests the trained Stage 2 policy's ability to handle wind disturbances.

UPDATED: Tests with 13 observations

Success criteria:
- 80%+ success rate with wind
- Can maintain hover despite gusts
- Average distance < 0.7m

Usage:
    python test_stage2_policy_v2.py --episodes 10
"""

import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_hover_disturbance_env_v2 import DroneHoverDisturbanceEnv
import argparse


def test_policy(model_path, vecnormalize_path, num_episodes=10, wind_strength=5.0):
    """Test the Stage 2 policy"""
    
    print("\n" + "="*70)
    print("🧪 TESTING STAGE 2: DISTURBANCE RECOVERY (13 OBSERVATIONS)")
    print("="*70)
    print("Testing neural network with WIND disturbances!")
    print()
    
    # Load model
    print(f"[1/3] Loading model: {model_path}")
    
    def make_env():
        env = DroneHoverDisturbanceEnv(
            target_altitude=10.0,
            max_steps=500,
            wind_strength=wind_strength,
            debug=False
        )
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path, env=env)
    
    print("   ✅ Model loaded")
    print("   📊 Observation space: 13")
    print()
    
    # Test
    print(f"[2/3] Running {num_episodes} test episodes...")
    print(f"   Wind strength: 0-{wind_strength} m/s")
    print(f"   Max steps: 500 per episode")
    print()
    print("="*70)
    
    results = []
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        
        episode_reward = 0
        distances = []
        wind_magnitudes = []
        max_wind = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            
            # Track metrics
            if len(info) > 0:
                distances.append(info[0].get('distance', 0))
                wind_mag = info[0].get('wind_magnitude', 0)
                wind_magnitudes.append(wind_mag)
                max_wind = max(max_wind, wind_mag)
            
            if done[0]:
                reason = info[0].get('reason', 'unknown')
                break
        else:
            reason = 'completed'
        
        # Results
        success = reason == 'completed' or reason == 'timeout'
        avg_distance = np.mean(distances) if distances else 0
        avg_wind = np.mean(wind_magnitudes) if wind_magnitudes else 0
        
        results.append({
            'episode': episode,
            'success': success,
            'steps': step + 1,
            'avg_distance': avg_distance,
            'avg_wind': avg_wind,
            'max_wind': max_wind,
            'reward': episode_reward,
            'reason': reason
        })
        
        # Print
        status = "✅" if success else "❌"
        print(f"Episode {episode:2d}/{num_episodes} | Steps: {step+1:3d} | "
              f"Success: {status} | Dist: {avg_distance:.2f}m | "
              f"Wind: {avg_wind:.1f}m/s (max: {max_wind:.1f}) | "
              f"Reason: {reason}")
    
    print("="*70)
    
    # Statistics
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_episodes * 100
    
    print(f"Success Rate: {success_rate:.0f}% ({successes}/{num_episodes} episodes)")
    
    if successes > 0:
        successful = [r for r in results if r['success']]
        avg_distance = np.mean([r['avg_distance'] for r in successful])
        avg_wind = np.mean([r['avg_wind'] for r in successful])
        max_wind_survived = np.max([r['max_wind'] for r in successful])
        
        print(f"Average Distance: {avg_distance:.2f}m (successful episodes)")
        print(f"Average Wind Handled: {avg_wind:.1f} m/s")
        print(f"Maximum Wind Survived: {max_wind_survived:.1f} m/s")
    
    avg_steps = np.mean([r['steps'] for r in results])
    print(f"Average Episode Length: {avg_steps:.1f} steps")
    
    print()
    
    # Failure analysis
    if successes < num_episodes:
        print("Failure Reasons:")
        reasons = {}
        for r in results:
            if not r['success']:
                reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        for reason, count in reasons.items():
            print(f"   - {reason}: {count}")
        print()
    
    # Verdict
    print("="*70)
    if success_rate >= 85:
        print("✅ EXCELLENT! Policy handles wind disturbances very well!")
        print("   Ready for Stage 3 (flip recovery)")
        print("   ✅ Model can be used for transfer learning to Stage 3!")
    elif success_rate >= 70:
        print("✅ GOOD! Policy handles wind reasonably well.")
        print("   Can proceed to Stage 3 or train longer")
    elif success_rate >= 50:
        print("⚠️  MODERATE! Policy partially learned disturbance recovery.")
        print("   Recommend: Train longer (750k timesteps)")
    else:
        print("❌ NEEDS IMPROVEMENT! Policy struggles with wind.")
        print("   Action needed:")
        print("   1. Train for more timesteps (1M)")
        print("   2. Reduce wind strength for initial training")
        print("   3. Check if Stage 1 policy loaded correctly")
    print("="*70 + "\n")
    
    # Comparison to Stage 1
    print("="*70)
    print("📊 COMPARISON TO STAGE 1")
    print("="*70)
    print(f"Stage 1 (no wind):  95%+ success, ~0.39m avg distance")
    print(f"Stage 2 (with wind): {success_rate:.0f}% success, {avg_distance:.2f}m avg distance")
    print()
    if success_rate >= 70:
        print("✅ Successfully maintained hover ability despite wind!")
    else:
        print("⚠️  Wind significantly impacted performance")
    print("="*70 + "\n")
    
    env.close()
    
    return results


def find_model_files(model_path, vecnormalize_path):
    """Auto-detect model files, including interrupted versions"""
    import os
    
    # Check if provided paths exist
    if os.path.exists(model_path) and os.path.exists(vecnormalize_path):
        return model_path, vecnormalize_path
    
    # Try interrupted versions
    model_interrupted = model_path.replace('.zip', '_interrupted.zip')
    vecnorm_interrupted = vecnormalize_path.replace('.pkl', '_interrupted.pkl')
    
    if os.path.exists(model_interrupted) and os.path.exists(vecnorm_interrupted):
        print(f"\n⚠️  Standard files not found. Using interrupted versions:")
        print(f"   Model: {model_interrupted}")
        print(f"   VecNormalize: {vecnorm_interrupted}\n")
        return model_interrupted, vecnorm_interrupted
    
    # Check for any checkpoint files
    checkpoint_dir = './models_v2/stage2_checkpoints/'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            vecnorm_checkpoint = checkpoint_path.replace('.zip', '_vecnormalize.pkl')
            
            if os.path.exists(checkpoint_path):
                print(f"\n⚠️  Standard files not found. Using latest checkpoint:")
                print(f"   Model: {checkpoint_path}")
                print(f"   VecNormalize: {vecnorm_checkpoint}\n")
                return checkpoint_path, vecnorm_checkpoint
    
    # If nothing found, print helpful error
    print("\n❌ ERROR: Could not find model files!")
    print("\nSearched for:")
    print(f"   1. {model_path}")
    print(f"   2. {model_interrupted}")
    print(f"   3. Checkpoints in {checkpoint_dir}")
    print("\nAvailable files in ./models_v2/:")
    if os.path.exists('./models_v2/'):
        for f in os.listdir('./models_v2/'):
            if f.endswith('.zip') or f.endswith('.pkl'):
                print(f"      - {f}")
    print("\nPlease specify the correct path using:")
    print("   python test_stage2_policy_v2.py --model <path> --vecnormalize <path>")
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Stage 2 disturbance recovery policy (13 observations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model files
  python test_stage2_policy_v2.py
  
  # Specify custom paths
  python test_stage2_policy_v2.py --model ./models_v2/my_model.zip --vecnormalize ./models_v2/my_vecnorm.pkl
  
  # Test with different settings
  python test_stage2_policy_v2.py --episodes 20 --wind-strength 3.0
        """
    )
    
    parser.add_argument('--model', type=str,
                        default='./models_v2/hover_disturbance_policy.zip',
                        help='Path to trained Stage 2 model (auto-detects _interrupted version)')
    parser.add_argument('--vecnormalize', type=str,
                        default='./models_v2/hover_disturbance_vecnormalize.pkl',
                        help='Path to VecNormalize stats (auto-detects _interrupted version)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes (default: 10)')
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Wind strength for testing in m/s (default: 5.0)')
    
    args = parser.parse_args()
    
    # Auto-detect model files
    model_path, vecnormalize_path = find_model_files(args.model, args.vecnormalize)
    
    results = test_policy(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        num_episodes=args.episodes,
        wind_strength=args.wind_strength
    )
