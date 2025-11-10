"""
STAGE 3 POLICY TESTING SCRIPT
==============================
Tests the trained Stage 3 flip recovery policy

Usage:
    python test_stage3_policy.py --episodes 20

    # Test with 100% disturbance (every episode)
python test_stage3_policy.py --flip-prob 1.0 --episodes 20

# Test with 50% disturbance
python test_stage3_policy.py --flip-prob 0.5 --episodes 20

# Test with 85% disturbance
python test_stage3_policy.py --flip-prob 0.85 --episodes 20

# Test with NO disturbances (0%)
python test_stage3_policy.py --flip-prob 0.0 --episodes 20

python test_stage3_policy.py \
  --model ./models/flip_recovery_policy_interrupted.zip \
  --vecnorm ./models/flip_recovery_vecnormalize_interrupted.pkl \
  --episodes 20 \
  --flip-prob 1.0


Bug fixes - 10/11/2025
BEFORE (BROKEN):
pythonenv = DroneFlipRecoveryEnv(
    target_altitude=10.0,   # ← Overriding to 10m!
    max_steps=500,          # ← Overriding to 500!
    
AFTER (FIXED):
pythonenv = DroneFlipRecoveryEnv(
    target_altitude=30.0,   # ← Using 30m!
    max_steps=600,          # ← Using 600!
Impact:

❌ Before: Only 9.5m recovery space (crashed easily)
✅ After: 29.5m recovery space (3x more!)

"""

import airsim
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_flip_recovery_env_injector import DroneFlipRecoveryEnv
import time


def test_policy(model_path, vecnorm_path, num_episodes=20, flip_prob=1.0):
    """
    Test the Stage 3 flip recovery policy
    
    Args:
        model_path: Path to trained model
        vecnorm_path: Path to VecNormalize stats
        num_episodes: Number of test episodes
        flip_prob: Probability of disturbance (1.0 = every episode)
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING STAGE 3: FLIP RECOVERY POLICY")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Disturbance probability: {flip_prob*100:.0f}%")
    print("="*70 + "\n")
    
    # Create environment
    print("[1/3] Creating test environment...")
    
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=30.0,  # ULTIMATE: 30m altitude!
            max_steps=600,         # ULTIMATE: 600 steps!
            wind_strength=5.0,
            flip_prob=flip_prob,
            debug=True
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    print(f"[2/3] Loading normalization stats...")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False  # Don't update stats during testing
    env.norm_reward = False
    
    print(f"[3/3] Loading trained model...")
    model = PPO.load(model_path)
    
    print("   ✅ Model loaded successfully!\n")
    
    # Test episodes
    results = {
        'successes': [],
        'distances': [],
        'episode_lengths': [],
        'reasons': [],
        'disturbance_initiated': [],
        'disturbance_recovered': [],
        'recovery_times': [],
        'max_angular_velocities': [],
    }
    
    print("="*70)
    print("🚀 STARTING TEST EPISODES")
    print("="*70 + "\n")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        max_ang_vel = 0
        
        print(f"Episode {episode+1}/{num_episodes}:")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_steps += 1
            
            # Track max angular velocity
            if len(obs[0]) >= 13:
                ang_vel = np.linalg.norm(obs[0][10:13])
                max_ang_vel = max(max_ang_vel, ang_vel)
            
            if done:
                break
        
        # Get info from environment
        env_info = info[0] if isinstance(info, list) else info
        
        # Determine success
        reason = env_info.get('reason', 'unknown')
        success = (reason == 'timeout')
        distance = env_info.get('distance', 999)
        
        disturbance_initiated = env_info.get('tumble_initiated', False)
        disturbance_recovered = env_info.get('tumble_recovered', False)
        recovery_steps = env_info.get('recovery_steps', 0)
        
        # Store results
        results['successes'].append(success)
        results['distances'].append(distance)
        results['episode_lengths'].append(episode_steps)
        results['reasons'].append(reason)
        results['disturbance_initiated'].append(disturbance_initiated)
        results['disturbance_recovered'].append(disturbance_recovered)
        results['recovery_times'].append(recovery_steps if disturbance_recovered else 0)
        results['max_angular_velocities'].append(max_ang_vel)
        
        # Print result
        status = "✅" if success else "💥"
        print(f"   {status} Steps: {episode_steps} | Dist: {distance:.2f}m | Reason: {reason}")
        if disturbance_initiated:
            recovery_status = "✅ Recovered" if disturbance_recovered else "❌ Crashed"
            print(f"      Disturbance: {recovery_status}")
            if disturbance_recovered:
                print(f"      Recovery time: {recovery_steps} steps ({recovery_steps*0.05:.1f}s)")
        print()
    
    # Calculate statistics
    print("="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    success_rate = np.mean(results['successes']) * 100
    avg_distance = np.mean([d for s, d in zip(results['successes'], results['distances']) if s])
    avg_length = np.mean(results['episode_lengths'])
    
    # Disturbance statistics
    num_disturbances = sum(results['disturbance_initiated'])
    num_recoveries = sum(results['disturbance_recovered'])
    recovery_rate = (num_recoveries / num_disturbances * 100) if num_disturbances > 0 else 0
    avg_recovery_time = np.mean([t for t in results['recovery_times'] if t > 0]) if num_recoveries > 0 else 0
    
    print(f"\n📈 Overall Performance:")
    print(f"   Success Rate: {success_rate:.0f}% ({sum(results['successes'])}/{num_episodes} episodes)")
    print(f"   Average Distance: {avg_distance:.2f}m (successful episodes)")
    print(f"   Average Episode Length: {avg_length:.1f} steps")
    
    print(f"\n🔄 Disturbance Recovery:")
    print(f"   Episodes with disturbance: {num_disturbances}/{num_episodes} ({num_disturbances/num_episodes*100:.0f}%)")
    print(f"   Successful recoveries: {num_recoveries}/{num_disturbances}")
    print(f"   Recovery Rate: {recovery_rate:.0f}%")
    if avg_recovery_time > 0:
        print(f"   Avg Recovery Time: {avg_recovery_time:.0f} steps ({avg_recovery_time*0.05:.1f}s)")
    
    print(f"\n📊 Episode Breakdown:")
    crash_reasons = {}
    for reason in results['reasons']:
        crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
    
    for reason, count in crash_reasons.items():
        print(f"   {reason}: {count} episodes")
    
    # Performance grading
    print(f"\n🎯 Performance Grade:")
    if recovery_rate >= 75:
        grade = "A+ (EXCELLENT!)"
    elif recovery_rate >= 65:
        grade = "A (VERY GOOD)"
    elif recovery_rate >= 55:
        grade = "B (GOOD)"
    elif recovery_rate >= 40:
        grade = "C (NEEDS IMPROVEMENT)"
    else:
        grade = "D (MORE TRAINING NEEDED)"
    
    print(f"   Recovery Performance: {grade}")
    
    if num_disturbances > 0:
        print(f"\n💡 Insights:")
        if recovery_rate < 50:
            print(f"   ⚠️  Recovery rate is low - continue training!")
        elif recovery_rate < 70:
            print(f"   📈 Good progress - train longer for better results")
        else:
            print(f"   ✅ Excellent recovery rate - ready for deployment!")
    
    print("\n" + "="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        default='./models/flip_recovery_policy_interrupted.zip',
                        help='Path to trained model')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/flip_recovery_vecnormalize_interrupted.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of test episodes')
    parser.add_argument('--flip-prob', type=float, default=1.0,
                        help='Disturbance probability (1.0 = every episode)')
    
    args = parser.parse_args()
    
    test_policy(args.model, args.vecnorm, args.episodes, args.flip_prob)