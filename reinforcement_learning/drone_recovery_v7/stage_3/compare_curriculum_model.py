"""
CURRICULUM LEVEL COMPARISON - PhD Analysis (FIXED)
===================================================
Tests each curriculum level model across ALL intensity ranges.

FIXED:
- Handles Monitor wrapper correctly
- Works with different environment names
- Better error messages

Usage:
    python compare_curriculum_model.py --episodes 20
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import argparse
from pathlib import Path
import json
from collections import defaultdict
import sys

# Try to import the environment (adjust name if needed)
def import_environment():
    """Try multiple environment import names"""
    env_names = [
        'drone_flip_recovery_env_injector_gated',
        'drone_flip_recovery_env_gated',
        'drone_recovery_env_gated'
    ]
    
    for env_name in env_names:
        try:
            module = __import__(env_name)
            # Try different class names
            for class_name in ['DroneFlipRecoveryEnv', 'DroneFlipRecoveryEnvGated', 'DroneRecoveryEnv']:
                if hasattr(module, class_name):
                    return getattr(module, class_name)
        except ImportError:
            continue
    
    print("❌ Error: Could not import environment!")
    print(f"   Tried: {', '.join(env_names)}")
    print("\n   Please ensure your environment file is in the same directory.")
    sys.exit(1)

DroneFlipRecoveryEnv = import_environment()


def get_base_env(vec_env):
    """Safely get base environment through wrapper layers"""
    env = vec_env.envs[0]
    
    # Unwrap Monitor if present
    if hasattr(env, 'env'):
        env = env.env
    
    # Unwrap any other wrappers
    while hasattr(env, 'env') and not hasattr(env, 'curriculum_level'):
        env = env.env
    
    return env


def test_model_at_intensity(model_path, vecnorm_path, intensity_range, num_episodes=20):
    """Test a model at specific intensity range"""
    
    print(f"   Loading model: {model_path}")
    
    # Create test environment
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=30.0,
            max_steps=600,
            wind_strength=0.0,  # Disable wind for controlled testing
            flip_prob=1.0,
            debug=False
        )
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    if Path(vecnorm_path).exists():
        print(f"   Loading VecNormalize: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"   ⚠️  VecNormalize not found: {vecnorm_path}")
        print(f"   ⚠️  Testing without normalization")
    
    # Load model
    model = PPO.load(model_path, env=env, device="cpu")
    print(f"   Model loaded successfully!")
    
    # Get base environment
    base_env = get_base_env(env)
    
    # Verify we can set curriculum level
    if not hasattr(base_env, 'curriculum_level'):
        print(f"   ⚠️  Warning: Environment doesn't have curriculum_level attribute")
        print(f"   ⚠️  Environment type: {type(base_env)}")
        print(f"   ⚠️  Testing may not work correctly")
    
    # Test episodes
    results = {
        'recoveries': 0,
        'attempts': 0,
        'recovery_times': [],
        'intensities': [],
        'disturbance_types': []
    }
    
    intensity_names = ['EASY (0.7-0.9×)', 'MEDIUM (0.9-1.1×)', 'HARD (1.1-1.5×)']
    print(f"   Testing at {intensity_names[intensity_range]}...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        # Set curriculum level for this test
        try:
            base_env = get_base_env(env)
            if hasattr(base_env, 'curriculum_level'):
                base_env.curriculum_level = intensity_range
        except Exception as e:
            if ep == 0:  # Only warn once
                print(f"   ⚠️  Could not set curriculum level: {e}")
        
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            if done:
                info_dict = info[0]
                if info_dict.get('tumble_initiated', False):
                    results['attempts'] += 1
                    results['intensities'].append(info_dict.get('disturbance_intensity', 1.0))
                    results['disturbance_types'].append(info_dict.get('disturbance_type', 'unknown'))
                    
                    if info_dict.get('tumble_recovered', False):
                        results['recoveries'] += 1
                        results['recovery_times'].append(info_dict.get('recovery_steps', 0))
                
                # Print progress
                if (ep + 1) % 5 == 0:
                    current_rate = (results['recoveries'] / results['attempts'] * 100) if results['attempts'] > 0 else 0
                    print(f"      Episode {ep+1}/{num_episodes}: {results['recoveries']}/{results['attempts']} recovered ({current_rate:.0f}%)")
    
    env.close()
    
    # Calculate statistics
    recovery_rate = (results['recoveries'] / results['attempts'] * 100) if results['attempts'] > 0 else 0
    avg_recovery_time = np.mean(results['recovery_times']) if results['recovery_times'] else 0
    avg_intensity = np.mean(results['intensities']) if results['intensities'] else 0
    
    print(f"   ✅ Complete: {recovery_rate:.1f}% recovery ({results['recoveries']}/{results['attempts']})\n")
    
    return {
        'recovery_rate': recovery_rate,
        'recoveries': results['recoveries'],
        'attempts': results['attempts'],
        'avg_recovery_time': avg_recovery_time,
        'avg_intensity': avg_intensity,
        'std_intensity': np.std(results['intensities']) if results['intensities'] else 0
    }


def main(args):
    print("\n" + "="*80)
    print("🎓 CURRICULUM LEVEL COMPARISON - PhD Analysis")
    print("="*80)
    print("Testing each curriculum level model across ALL intensity ranges")
    print("This demonstrates:")
    print("  1. Progressive improvement through curriculum learning")
    print("  2. No catastrophic forgetting (still works on easier cases)")
    print("  3. Generalization capabilities")
    print("="*80 + "\n")
    
    # Define models to test
    models = {
        'Level 0 (EASY)': {
            'model': './models/stage3_checkpoints/curriculum_levels/level_0_EASY_mastered.zip',
            'vecnorm': './models/stage3_checkpoints/curriculum_levels/level_0_EASY_mastered_vecnormalize.pkl',
            'trained_on': 'EASY (0.7-0.9×)'
        },
        'Level 1 (MEDIUM)': {
            'model': './models/stage3_checkpoints/curriculum_levels/level_1_MEDIUM_mastered.zip',
            'vecnorm': './models/stage3_checkpoints/curriculum_levels/level_1_MEDIUM_mastered_vecnormalize.pkl',
            'trained_on': 'MEDIUM (0.9-1.1×)'
        },
        'Final Model': {
            'model': './models/stage3_checkpoints/gated_curriculum_policy.zip',
            'vecnorm': './models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl',
            'trained_on': 'HARD (1.1-1.5×)'
        }
    }
    
    # Test intensity ranges
    intensity_ranges = {
        'EASY': 0,      # 0.7-0.9×
        'MEDIUM': 1,    # 0.9-1.1×
        'HARD': 2       # 1.1-1.5×
    }
    
    # Store all results
    all_results = defaultdict(dict)
    
    # Test each model
    for model_name, model_info in models.items():
        if not Path(model_info['model']).exists():
            print(f"⚠️  {model_name} not found: {model_info['model']}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📊 Testing: {model_name}")
        print(f"   Trained on: {model_info['trained_on']}")
        print(f"   Model: {model_info['model']}")
        print(f"{'='*80}\n")
        
        # Test at each intensity
        for intensity_name, intensity_level in intensity_ranges.items():
            try:
                results = test_model_at_intensity(
                    model_info['model'],
                    model_info['vecnorm'],
                    intensity_level,
                    num_episodes=args.episodes
                )
                
                all_results[model_name][intensity_name] = results
                
            except Exception as e:
                print(f"   ❌ Error testing {intensity_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print comprehensive comparison table
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS - PhD Table 5.3")
    print("="*80)
    print("Cross-Level Performance Evaluation: No Catastrophic Forgetting")
    print("-"*80)
    
    # Table header
    print(f"\n{'Model':<25} {'Trained On':<20} {'Test Intensity':<15} {'Recovery Rate':<20} {'Avg Time (steps)':<15}")
    print("-"*80)
    
    for model_name in models.keys():
        if model_name not in all_results:
            continue
        
        model_info = models[model_name]
        for i, (intensity_name, results) in enumerate(all_results[model_name].items()):
            model_col = model_name if i == 0 else ""
            trained_col = model_info['trained_on'] if i == 0 else ""
            
            recovery_str = f"{results['recovery_rate']:.1f}% ({results['recoveries']}/{results['attempts']})"
            time_str = f"{results['avg_recovery_time']:.1f}" if results['avg_recovery_time'] > 0 else "N/A"
            
            print(f"{model_col:<25} {trained_col:<20} {intensity_name:<15} {recovery_str:<20} {time_str:<15}")
        
        if model_name != list(models.keys())[-1]:
            print("-"*80)
    
    print("="*80 + "\n")
    
    # Save results to JSON
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert defaultdict to regular dict for JSON serialization
    results_dict = {
        'models_tested': list(models.keys()),
        'test_episodes_per_intensity': args.episodes,
        'results': {k: dict(v) for k, v in all_results.items()}
    }
    
    output_path = output_dir / "curriculum_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    print(f"\n✅ Analysis Complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of test episodes per intensity level')
    
    args = parser.parse_args()
    main(args)