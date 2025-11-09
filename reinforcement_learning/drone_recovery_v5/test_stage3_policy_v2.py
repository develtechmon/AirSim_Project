"""
TEST STAGE 3: FLIP RECOVERY POLICY (13 OBSERVATIONS)
=====================================================
Tests the trained Stage 3 policy's ability to recover from flips.

Success criteria:
- 70%+ recovery rate from flips
- Recovery time < 200 steps (10 seconds)
- Maintain hover after recovery
- Handle flips + wind simultaneously

Usage:
    python test_stage3_policy_v2.py --episodes 20
"""

import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append('../stage2_v2')
from drone_flip_recovery_env import DroneFlipRecoveryEnv
import argparse


def test_policy(model_path, vecnormalize_path, num_episodes=10, 
                wind_strength=5.0, flip_prob=1.0):
    """Test the Stage 3 flip recovery policy"""
    
    print("\n" + "="*70)
    print("🔄 TESTING STAGE 3: FLIP RECOVERY (13 OBSERVATIONS)")
    print("="*70)
    print("Testing neural network with FLIPS + WIND!")
    print()
    
    # Load model
    print(f"[1/3] Loading model: {model_path}")
    
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=10.0,
            max_steps=500,
            wind_strength=wind_strength,
            flip_prob=flip_prob,
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
    print(f"   Flip probability: {flip_prob*100:.0f}%")
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
        was_flipped = False
        recovered = False
        recovery_time = 0
        
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
                
                # Track flip/recovery
                if info[0].get('is_flipped', False):
                    was_flipped = True
                if info[0].get('has_recovered', False):
                    recovered = True
                    recovery_time = info[0].get('recovery_steps', 0)
            
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
            'was_flipped': was_flipped,
            'recovered': recovered,
            'recovery_time': recovery_time,
            'reason': reason
        })
        
        # Print
        status = "✅" if success else "❌"
        flip_info = ""
        if was_flipped:
            if recovered:
                flip_info = f"Recovered in {recovery_time} steps"
            else:
                flip_info = "Failed to recover"
        else:
            flip_info = "Started upright"
        
        print(f"Episode {episode:2d}/{num_episodes} | Steps: {step+1:3d} | "
              f"Success: {status} | Dist: {avg_distance:.2f}m | "
              f"{flip_info}")
    
    print("="*70)
    
    # Statistics
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_episodes * 100
    
    flipped_episodes = [r for r in results if r['was_flipped']]
    recovered_episodes = [r for r in flipped_episodes if r['recovered']]
    
    print(f"Overall Success Rate: {success_rate:.0f}% ({successes}/{num_episodes} episodes)")
    
    if len(flipped_episodes) > 0:
        recovery_rate = len(recovered_episodes) / len(flipped_episodes) * 100
        print(f"\n🔄 FLIP RECOVERY:")
        print(f"   Flipped Episodes: {len(flipped_episodes)}")
        print(f"   Recovery Rate: {recovery_rate:.0f}% ({len(recovered_episodes)}/{len(flipped_episodes)})")
        
        if len(recovered_episodes) > 0:
            avg_recovery_time = np.mean([r['recovery_time'] for r in recovered_episodes])
            print(f"   Avg Recovery Time: {avg_recovery_time:.0f} steps ({avg_recovery_time*0.05:.1f} seconds)")
            
            # Stats for recovered episodes
            recovered_successful = [r for r in recovered_episodes if r['success']]
            if len(recovered_successful) > 0:
                avg_distance = np.mean([r['avg_distance'] for r in recovered_successful])
                print(f"   Avg Distance After Recovery: {avg_distance:.2f}m")
    
    if successes > 0:
        successful = [r for r in results if r['success']]
        avg_distance = np.mean([r['avg_distance'] for r in successful])
        avg_wind = np.mean([r['avg_wind'] for r in successful])
        max_wind_survived = np.max([r['max_wind'] for r in successful])
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Average Distance: {avg_distance:.2f}m")
        print(f"   Average Wind Handled: {avg_wind:.1f} m/s")
        print(f"   Maximum Wind Survived: {max_wind_survived:.1f} m/s")
    
    avg_steps = np.mean([r['steps'] for r in results])
    print(f"   Average Episode Length: {avg_steps:.1f} steps")
    
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
    if success_rate >= 80 and (len(flipped_episodes) == 0 or recovery_rate >= 70):
        print("✅ EXCELLENT! Policy handles flips and disturbances very well!")
        print("   🎉 Stage 3 COMPLETE!")
        print("   🎉 3-STAGE CURRICULUM MASTERED!")
    elif success_rate >= 70 and (len(flipped_episodes) == 0 or recovery_rate >= 60):
        print("✅ GOOD! Policy handles flip recovery reasonably well.")
        print("   Can be used or train longer for better results")
    elif success_rate >= 50:
        print("⚠️  MODERATE! Policy partially learned flip recovery.")
        print("   Recommend: Train longer (500k timesteps)")
    else:
        print("❌ NEEDS IMPROVEMENT! Policy struggles with flip recovery.")
        print("   Action needed:")
        print("   1. Train for more timesteps (500k)")
        print("   2. Reduce flip probability initially (0.3)")
        print("   3. Check if Stage 2 policy loaded correctly")
    print("="*70 + "\n")
    
    # Comparison to previous stages
    print("="*70)
    print("📊 COMPARISON ACROSS ALL STAGES")
    print("="*70)
    print(f"Stage 1 (hover):              95%+ success, 0.39m avg distance")
    print(f"Stage 2 (wind):               90%+ success, 0.48m avg distance")
    print(f"Stage 3 (wind + flips):       {success_rate:.0f}% success, {avg_distance:.2f}m avg distance")
    print()
    if len(flipped_episodes) > 0:
        print(f"Flip Recovery Rate: {recovery_rate:.0f}%")
        if recovery_rate >= 70:
            print("✅ Successfully learned flip recovery!")
        else:
            print("⚠️  Flip recovery needs improvement")
    print("="*70 + "\n")
    
    env.close()
    
    return results


def find_model_files(model_path, vecnormalize_path):
    """Auto-detect model files"""
    import os
    
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
    
    # Check for checkpoints
    checkpoint_dir = './models_v2/stage3_checkpoints/'
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
    
    # If nothing found
    print("\n❌ ERROR: Could not find model files!")
    print("\nSearched for:")
    print(f"   1. {model_path}")
    print(f"   2. {model_interrupted}")
    print(f"   3. Checkpoints in {checkpoint_dir}")
    print("\nPlease train Stage 3 first using:")
    print("   python train_stage3_flip_v2.py")
    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Stage 3 flip recovery policy (13 observations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model files
  python test_stage3_policy_v2.py
  
  # Test with more episodes
  python test_stage3_policy_v2.py --episodes 20
  
  # Test only flipped starts (100% flip probability)
  python test_stage3_policy_v2.py --flip-prob 1.0 --episodes 20
  
  # Test without flips (just wind handling)
  python test_stage3_policy_v2.py --flip-prob 0.0
        """
    )
    
    parser.add_argument('--model', type=str,
                        default='./models/flip_recovery_policy.zip',
                        help='Path to trained Stage 3 model')
    parser.add_argument('--vecnormalize', type=str,
                        default='./models/flip_recovery_vecnormalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes (default: 10)')
    parser.add_argument('--wind-strength', type=float, default=5.0,
                        help='Wind strength for testing in m/s (default: 5.0)')
    parser.add_argument('--flip-prob', type=float, default=1.0,
                        help='Probability of starting flipped (default: 1.0 = always flipped)')
    
    args = parser.parse_args()
    
    # Auto-detect model files
    model_path, vecnormalize_path = find_model_files(args.model, args.vecnormalize)
    
    results = test_policy(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        num_episodes=args.episodes,
        wind_strength=args.wind_strength,
        flip_prob=args.flip_prob
    )
