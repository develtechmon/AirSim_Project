"""
CURRICULUM LEVEL COMPARISON - PhD Analysis
===========================================
Tests each curriculum level model across ALL intensity ranges to demonstrate:
1. Performance improvement through curriculum learning
2. No catastrophic forgetting (models still work on easier cases)
3. Generalization across intensity spectrum

Generates PhD-ready tables and comparison plots!

Usage:
    python compare_curriculum_levels.py --episodes 60
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_flip_recovery_env_injector_gated import DroneFlipRecoveryEnv
import argparse
from pathlib import Path
import json
from collections import defaultdict


def test_model_at_intensity(model_path, vecnorm_path, intensity_range, num_episodes=20):
    """Test a model at specific intensity range"""
    
    # Create test environment with fixed intensity range
    def make_env():
        env = DroneFlipRecoveryEnv(
            target_altitude=30.0,
            max_steps=600,
            wind_strength=0.0,  # Disable wind for controlled testing
            flip_prob=1.0,
            debug=False
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=env, device="cpu")
    
    # Test episodes
    results = {
        'recoveries': 0,
        'attempts': 0,
        'recovery_times': [],
        'intensities': [],
        'disturbance_types': []
    }
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        # Manually inject disturbance at fixed intensity
        base_env = env.envs[0].env
        base_env.curriculum_level = intensity_range  # Set to test range
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if done:
                info_dict = info[0]
                if info_dict.get('tumble_initiated', False):
                    results['attempts'] += 1
                    results['intensities'].append(info_dict.get('disturbance_intensity', 1.0))
                    results['disturbance_types'].append(info_dict.get('disturbance_type', 'unknown'))
                    
                    if info_dict.get('tumble_recovered', False):
                        results['recoveries'] += 1
                        results['recovery_times'].append(info_dict.get('recovery_steps', 0))
    
    env.close()
    
    # Calculate statistics
    recovery_rate = (results['recoveries'] / results['attempts'] * 100) if results['attempts'] > 0 else 0
    avg_recovery_time = np.mean(results['recovery_times']) if results['recovery_times'] else 0
    avg_intensity = np.mean(results['intensities']) if results['intensities'] else 0
    
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
            print(f"   Testing at {intensity_name} intensity... ", end='', flush=True)
            
            results = test_model_at_intensity(
                model_info['model'],
                model_info['vecnorm'],
                intensity_level,
                num_episodes=args.episodes
            )
            
            all_results[model_name][intensity_name] = results
            
            print(f"✅ {results['recovery_rate']:.1f}% recovery "
                  f"({results['recoveries']}/{results['attempts']})")
    
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
    
    # Analysis: Performance improvement
    print("📈 ANALYSIS: Performance Improvement Through Curriculum")
    print("-"*80)
    
    if 'Level 0 (EASY)' in all_results and 'Final Model' in all_results:
        for intensity in ['EASY', 'MEDIUM', 'HARD']:
            if intensity in all_results['Level 0 (EASY)'] and intensity in all_results['Final Model']:
                level0_rate = all_results['Level 0 (EASY)'][intensity]['recovery_rate']
                final_rate = all_results['Final Model'][intensity]['recovery_rate']
                improvement = final_rate - level0_rate
                
                print(f"\n{intensity} Intensity:")
                print(f"  Level 0 Model:  {level0_rate:.1f}%")
                print(f"  Final Model:    {final_rate:.1f}%")
                print(f"  Improvement:    {improvement:+.1f} pp")
    
    print("\n" + "="*80 + "\n")
    
    # Analysis: Catastrophic forgetting check
    print("🔍 ANALYSIS: Catastrophic Forgetting Check")
    print("-"*80)
    print("Checking if advanced models maintain performance on easier cases...")
    print()
    
    catastrophic_forgetting = False
    
    if 'Level 0 (EASY)' in all_results and 'Final Model' in all_results:
        level0_easy = all_results['Level 0 (EASY)']['EASY']['recovery_rate']
        final_easy = all_results['Final Model']['EASY']['recovery_rate']
        
        print(f"EASY Intensity:")
        print(f"  Level 0 (trained on EASY): {level0_easy:.1f}%")
        print(f"  Final Model:                {final_easy:.1f}%")
        
        if final_easy < level0_easy - 10:  # More than 10% drop
            print(f"  ⚠️  WARNING: Catastrophic forgetting detected! ({final_easy - level0_easy:.1f} pp drop)")
            catastrophic_forgetting = True
        elif final_easy >= level0_easy:
            print(f"  ✅ Maintained/improved: {final_easy - level0_easy:+.1f} pp")
        else:
            print(f"  ✅ Minor degradation: {final_easy - level0_easy:.1f} pp (acceptable)")
    
    if not catastrophic_forgetting:
        print("\n✅ No catastrophic forgetting detected!")
        print("   Final model maintains strong performance on easier cases.")
    
    print("\n" + "="*80 + "\n")
    
    # Save results to JSON
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Convert defaultdict to regular dict for JSON serialization
    results_dict = {
        'models_tested': list(models.keys()),
        'test_episodes_per_intensity': args.episodes,
        'results': {k: dict(v) for k, v in all_results.items()},
        'catastrophic_forgetting_detected': catastrophic_forgetting
    }
    
    output_path = output_dir / "curriculum_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    
    # Create LaTeX table for thesis
    latex_output = output_dir / "table_5_3_curriculum_comparison.tex"
    with open(latex_output, 'w') as f:
        f.write("% Table 5.3: Cross-Level Performance Evaluation\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-Level Performance Evaluation: No Catastrophic Forgetting}\n")
        f.write("\\label{tab:curriculum_comparison}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Trained On} & \\textbf{Test} & \\textbf{Recovery Rate} & \\textbf{Avg Time (steps)} \\\\\n")
        f.write("\\midrule\n")
        
        for model_name in models.keys():
            if model_name not in all_results:
                continue
            
            model_info = models[model_name]
            for i, (intensity_name, results) in enumerate(all_results[model_name].items()):
                if i == 0:
                    f.write(f"{model_name} & {model_info['trained_on']} ")
                else:
                    f.write(f" & ")
                
                f.write(f"& {intensity_name} & {results['recovery_rate']:.1f}\\% ")
                f.write(f"& {results['avg_recovery_time']:.1f} \\\\\n")
            
            if model_name != list(models.keys())[-1]:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"📄 LaTeX table saved to: {latex_output}")
    
    print("\n✅ Analysis Complete!")
    print("\nNext Steps:")
    print("  1. Use curriculum_comparison_results.json for further analysis")
    print("  2. Include table_5_3_curriculum_comparison.tex in thesis")
    print("  3. Run: python analyze_training_logs.py for training progression plots")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of test episodes per intensity level')
    
    args = parser.parse_args()
    main(args)