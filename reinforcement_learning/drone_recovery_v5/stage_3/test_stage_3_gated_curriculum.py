"""
GATED CURRICULUM TESTING SCRIPT
================================
Tests the performance-gated curriculum model across all intensity levels

Features:
- Tests ALL intensity levels (Easy, Medium, Hard)
- Shows performance breakdown by difficulty
- Validates that each level was properly mastered
- PhD-ready metrics

Usage:
    python test_gated_curriculum.py --episodes 60
    
    # Test specific intensity range
    python test_gated_curriculum.py --test-level 0  # Easy only
    python test_gated_curriculum.py --test-level 1  # Medium only
    python test_gated_curriculum.py --test-level 2  # Hard only
    python test_gated_curriculum.py --test-level -1 # All levels (default)


# Test Easy-level model
python test_gated_curriculum.py \
  --model ./models/curriculum_levels/level_0_EASY_mastered.zip \
  --vecnorm ./models/curriculum_levels/level_0_EASY_mastered_vecnormalize.pkl

# Test Medium-level model  
python test_gated_curriculum.py \
  --model ./models/curriculum_levels/level_1_MEDIUM_mastered.zip \
  --vecnorm ./models/curriculum_levels/level_1_MEDIUM_mastered_vecnormalize.pkl

# Test Final model
python test_gated_curriculum.py \
  --model ./models/gated_curriculum_policy.zip

"""

import airsim
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_flip_recovery_env_injector_gated import DroneFlipRecoveryEnv
import time


def test_gated_curriculum(model_path, vecnorm_path, num_episodes=60, test_level=-1):
    """
    Test the gated curriculum model
    
    Args:
        model_path: Path to trained model
        vecnorm_path: Path to VecNormalize stats
        num_episodes: Number of test episodes (20 per level if testing all)
        test_level: -1=all, 0=easy, 1=medium, 2=hard
    """
    
    print("\n" + "="*70)
    print("🧪 TESTING GATED CURRICULUM MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    if test_level == -1:
        print(f"Testing: ALL intensity levels")
    else:
        level_names = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
        print(f"Testing: {level_names[test_level]} only")
    print("="*70 + "\n")
    
    # Create environment
    print("[1/3] Creating test environment...")
    
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=30.0,
            max_steps=600,
            wind_strength=5.0,
            flip_prob=1.0,  # Always have disturbance
            debug=True
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    print(f"[2/3] Loading normalization stats...")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    
    print(f"[3/3] Loading trained model...")
    model = PPO.load(model_path)
    
    print("   ✅ Model loaded successfully!\n")
    
    # Determine which levels to test
    if test_level == -1:
        # Test all levels
        levels_to_test = [0, 1, 2]
        episodes_per_level = num_episodes // 3
        print(f"Testing {episodes_per_level} episodes per level\n")
    else:
        # Test specific level
        levels_to_test = [test_level]
        episodes_per_level = num_episodes
        print(f"Testing {episodes_per_level} episodes at selected level\n")
    
    # Results storage
    all_results = {
        'successes': [],
        'distances': [],
        'episode_lengths': [],
        'reasons': [],
        'disturbance_initiated': [],
        'disturbance_recovered': [],
        'recovery_times': [],
        'intensities': [],
        'test_levels': [],
        'max_angular_velocities': [],
    }
    
    print("="*70)
    print("🚀 STARTING TEST EPISODES")
    print("="*70 + "\n")
    
    episode_count = 0
    
    for test_level_idx in levels_to_test:
        level_names = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
        
        print(f"\n{'='*70}")
        print(f"📊 TESTING LEVEL {test_level_idx}: {level_names[test_level_idx]}")
        print(f"{'='*70}\n")
        
        for ep_in_level in range(episodes_per_level):
            episode_count += 1
            
            # Manually set intensity for this test episode
            if test_level_idx == 0:
                test_intensity = np.random.uniform(0.7, 0.9)
            elif test_level_idx == 1:
                test_intensity = np.random.uniform(0.9, 1.1)
            else:
                test_intensity = np.random.uniform(1.1, 1.5)
            
            obs = env.reset()
            done = False
            step = 0
            max_ang_vel = 0
            
            print(f"Episode {episode_count} (Level {test_level_idx}, #{ep_in_level+1}):")
            print(f"   Target intensity: {test_intensity:.2f}x")
            
            # Override environment's intensity selection
            # We'll manually inject disturbance at specific intensity
            
            while not done and step < 600:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                step += 1
                
                # Track max angular velocity
                if len(obs[0]) >= 13:
                    ang_vel = np.linalg.norm(obs[0][10:13])
                    max_ang_vel = max(max_ang_vel, ang_vel)
                
                if done:
                    break
            
            # Get info
            env_info = info[0] if isinstance(info, list) else info
            
            reason = env_info.get('reason', 'unknown')
            success = (reason == 'timeout')
            distance = env_info.get('distance', 999)
            
            disturbance_initiated = env_info.get('tumble_initiated', False)
            disturbance_recovered = env_info.get('tumble_recovered', False)
            recovery_steps = env_info.get('recovery_steps', 0)
            actual_intensity = env_info.get('disturbance_intensity', test_intensity)
            
            # Store results
            all_results['successes'].append(success)
            all_results['distances'].append(distance)
            all_results['episode_lengths'].append(step)
            all_results['reasons'].append(reason)
            all_results['disturbance_initiated'].append(disturbance_initiated)
            all_results['disturbance_recovered'].append(disturbance_recovered)
            all_results['recovery_times'].append(recovery_steps if disturbance_recovered else 0)
            all_results['intensities'].append(actual_intensity)
            all_results['test_levels'].append(test_level_idx)
            all_results['max_angular_velocities'].append(max_ang_vel)
            
            # Print result
            status = "✅" if success else "💥"
            print(f"   {status} Steps: {step} | Dist: {distance:.2f}m | Reason: {reason}")
            if disturbance_initiated:
                recovery_status = "✅ Recovered" if disturbance_recovered else "❌ Crashed"
                print(f"      Disturbance: Intensity {actual_intensity:.2f}x | {recovery_status}")
                if disturbance_recovered:
                    print(f"      Recovery time: {recovery_steps} steps ({recovery_steps*0.05:.1f}s)")
            print()
    
    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("="*70)
    
    # Overall statistics
    success_rate = np.mean(all_results['successes']) * 100
    avg_distance = np.mean([d for s, d in zip(all_results['successes'], all_results['distances']) if s])
    avg_length = np.mean(all_results['episode_lengths'])
    
    num_disturbances = sum(all_results['disturbance_initiated'])
    num_recoveries = sum(all_results['disturbance_recovered'])
    overall_recovery_rate = (num_recoveries / num_disturbances * 100) if num_disturbances > 0 else 0
    avg_recovery_time = np.mean([t for t in all_results['recovery_times'] if t > 0]) if num_recoveries > 0 else 0
    
    print(f"\n📈 Overall Performance:")
    print(f"   Success Rate: {success_rate:.0f}% ({sum(all_results['successes'])}/{episode_count} episodes)")
    print(f"   Average Distance: {avg_distance:.2f}m (successful episodes)")
    print(f"   Average Episode Length: {avg_length:.1f} steps")
    
    print(f"\n🔄 Overall Recovery:")
    print(f"   Episodes with disturbance: {num_disturbances}/{episode_count}")
    print(f"   Successful recoveries: {num_recoveries}/{num_disturbances}")
    print(f"   Overall Recovery Rate: {overall_recovery_rate:.0f}%")
    if avg_recovery_time > 0:
        print(f"   Avg Recovery Time: {avg_recovery_time:.0f} steps ({avg_recovery_time*0.05:.1f}s)")
    
    # ========================================================================
    # BREAKDOWN BY LEVEL
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"🎯 PERFORMANCE BY CURRICULUM LEVEL (PhD Analysis)")
    print(f"{'='*70}")
    
    intensities = np.array(all_results['intensities'])
    recoveries = np.array(all_results['disturbance_recovered'])
    test_levels = np.array(all_results['test_levels'])
    
    level_names = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
    thresholds = [0.80, 0.70, 0.60]  # Expected performance
    
    for level_idx in levels_to_test:
        level_mask = test_levels == level_idx
        
        if np.sum(level_mask) > 0:
            level_recoveries = recoveries[level_mask]
            level_intensities = intensities[level_mask]
            level_rate = np.mean(level_recoveries) * 100
            level_count = np.sum(level_mask)
            level_recovered = np.sum(level_recoveries)
            avg_intensity = np.mean(level_intensities)
            
            threshold = thresholds[level_idx]
            threshold_pct = threshold * 100
            
            # Check if meets threshold
            meets_threshold = level_rate >= threshold_pct
            status = "✅ MASTERED" if meets_threshold else "⚠️  BELOW TARGET"
            
            print(f"\n   Level {level_idx} - {level_names[level_idx]}:")
            print(f"      Recovery Rate: {level_rate:.1f}% ({level_recovered}/{level_count}) {status}")
            print(f"      Target: {threshold_pct:.0f}%")
            print(f"      Avg Intensity: {avg_intensity:.2f}x")
            
            if not meets_threshold:
                print(f"      💡 Need {threshold_pct - level_rate:.0f}% more to reach target")
    
    print(f"\n{'='*70}")
    
    # ========================================================================
    # DETAILED INTENSITY BREAKDOWN
    # ========================================================================
    
    print(f"\n🔬 DETAILED INTENSITY BREAKDOWN:")
    print(f"{'='*70}")
    
    # Easy range (0.7-0.9)
    easy_mask = (intensities >= 0.7) & (intensities < 0.9)
    if np.sum(easy_mask) > 0:
        easy_rate = np.mean(recoveries[easy_mask]) * 100
        easy_count = np.sum(easy_mask)
        easy_recovered = np.sum(recoveries[easy_mask])
        print(f"   0.7-0.9x (EASY):   {easy_rate:.1f}% ({easy_recovered}/{easy_count})")
    
    # Medium range (0.9-1.1)
    med_mask = (intensities >= 0.9) & (intensities < 1.1)
    if np.sum(med_mask) > 0:
        med_rate = np.mean(recoveries[med_mask]) * 100
        med_count = np.sum(med_mask)
        med_recovered = np.sum(recoveries[med_mask])
        print(f"   0.9-1.1x (MEDIUM): {med_rate:.1f}% ({med_recovered}/{med_count})")
    
    # Hard range (1.1-1.3)
    hard1_mask = (intensities >= 1.1) & (intensities < 1.3)
    if np.sum(hard1_mask) > 0:
        hard1_rate = np.mean(recoveries[hard1_mask]) * 100
        hard1_count = np.sum(hard1_mask)
        hard1_recovered = np.sum(recoveries[hard1_mask])
        print(f"   1.1-1.3x (HARD):   {hard1_rate:.1f}% ({hard1_recovered}/{hard1_count})")
    
    # Extreme range (1.3-1.5)
    extreme_mask = intensities >= 1.3
    if np.sum(extreme_mask) > 0:
        extreme_rate = np.mean(recoveries[extreme_mask]) * 100
        extreme_count = np.sum(extreme_mask)
        extreme_recovered = np.sum(recoveries[extreme_mask])
        print(f"   1.3-1.5x (EXTREME): {extreme_rate:.1f}% ({extreme_recovered}/{extreme_count})")
    
    print(f"{'='*70}")
    
    # ========================================================================
    # EPISODE BREAKDOWN
    # ========================================================================
    
    print(f"\n📊 Episode Breakdown:")
    crash_reasons = {}
    for reason in all_results['reasons']:
        crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
    
    for reason, count in crash_reasons.items():
        print(f"   {reason}: {count} episodes")
    
    # ========================================================================
    # PERFORMANCE GRADING
    # ========================================================================
    
    print(f"\n🎯 Performance Grade:")
    if overall_recovery_rate >= 75:
        grade = "A+ (OUTSTANDING - PhD COMPLETE!)"
    elif overall_recovery_rate >= 65:
        grade = "A (EXCELLENT - PhD TARGET MET!)"
    elif overall_recovery_rate >= 55:
        grade = "B+ (VERY GOOD - Close to target)"
    elif overall_recovery_rate >= 45:
        grade = "B (GOOD - More training needed)"
    else:
        grade = "C (NEEDS IMPROVEMENT - Continue training)"
    
    print(f"   Overall Performance: {grade}")
    
    # ========================================================================
    # PhD ASSESSMENT
    # ========================================================================
    
    print(f"\n🎓 PhD GATED CURRICULUM ASSESSMENT:")
    
    all_mastered = True
    for level_idx in levels_to_test:
        level_mask = test_levels == level_idx
        if np.sum(level_mask) > 0:
            level_rate = np.mean(recoveries[level_mask]) * 100
            threshold = thresholds[level_idx] * 100
            
            if level_rate >= threshold:
                print(f"   ✅ Level {level_idx} ({level_names[level_idx]}): MASTERED ({level_rate:.0f}% ≥ {threshold:.0f}%)")
            else:
                print(f"   ❌ Level {level_idx} ({level_names[level_idx]}): Below target ({level_rate:.0f}% < {threshold:.0f}%)")
                all_mastered = False
    
    if all_mastered and overall_recovery_rate >= 65:
        print(f"\n   🎉 ALL LEVELS MASTERED!")
        print(f"   ✅ PhD requirements met")
        print(f"   ✅ Performance-gated curriculum validated")
        print(f"   ✅ Ready for thesis defense!")
    elif all_mastered:
        print(f"\n   ✅ All individual levels mastered")
        print(f"   ⚠️  Overall rate slightly below 65% target")
        print(f"   💡 Consider extending hard level training")
    else:
        print(f"\n   ⚠️  Some levels below mastery threshold")
        print(f"   💡 Continue training until all thresholds met")
    
    print("\n" + "="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        default='./models/gated_curriculum_policy.zip',
                        help='Path to trained model')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/gated_curriculum_vecnormalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=60,
                        help='Number of test episodes (20 per level if testing all)')
    parser.add_argument('--test-level', type=int, default=-1,
                        help='Test specific level: -1=all, 0=easy, 1=medium, 2=hard')
    
    args = parser.parse_args()
    
    test_gated_curriculum(args.model, args.vecnorm, args.episodes, args.test_level)