"""
TRAINING LOG ANALYSIS AND VISUALIZATION
========================================
Analyzes training logs and generates PhD-quality plots!

Generates:
    - Learning curves (reward over time)
    - Recovery rate progression
    - Curriculum advancement visualization
    - Statistics tables for thesis

Usage:
    # Auto-detect latest log (recommended):
    python analyze_training_logs.py --output-dir ./plots
    
    # Or specify exact file:
    python analyze_training_logs.py --log logs/stage3/gated_training_TIMESTAMP_episodes.csv --output-dir ./plots


    python analyze_training_logs.py --output-dir ./thesis_figures (i'm using this command to generate the figure)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path


def load_training_data(csv_path):
    """Load training data from CSV"""
    print(f"\n📂 Loading training data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"   ✅ Loaded {len(df)} episodes")
    print(f"   📊 Columns: {', '.join(df.columns)}")
    
    return df


def plot_learning_curves(df, output_dir):
    """Generate learning curve plots"""
    print(f"\n📈 Generating learning curves...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style for PhD-quality plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progression: Performance-Gated Curriculum Learning', 
                 fontsize=14, fontweight='bold')
    
    # 1. Episode Reward
    ax = axes[0, 0]
    ax.plot(df['episode'], df['episode_reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax.plot(df['episode'], df['rolling_10_reward'], linewidth=2, color='darkblue', label='Rolling 10 Avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Recovery Rate
    ax = axes[0, 1]
    ax.plot(df['episode'], df['rolling_10_recovery_rate'], linewidth=2, color='green', label='Rolling 10')
    ax.plot(df['episode'], df['rolling_50_recovery_rate'], linewidth=2, color='darkgreen', label='Rolling 50')
    ax.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Level 0 Target (80%)')
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Level 1 Target (70%)')
    ax.axhline(y=60, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='Level 2 Target (60%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Recovery Rate Progression')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Curriculum Level
    ax = axes[1, 0]
    ax.plot(df['episode'], df['curriculum_level'], linewidth=2, color='purple')
    ax.fill_between(df['episode'], df['curriculum_level'], alpha=0.3, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Curriculum Level')
    ax.set_title('Curriculum Progression')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['EASY\n(0.7-0.9×)', 'MEDIUM\n(0.9-1.1×)', 'HARD\n(1.1-1.5×)'])
    ax.grid(True, alpha=0.3)
    
    # 4. Disturbance Intensity
    ax = axes[1, 1]
    # Filter only episodes with disturbances
    df_disturbed = df[df['tumble_initiated'] == 1].copy()
    if len(df_disturbed) > 0:
        ax.scatter(df_disturbed['episode'], df_disturbed['disturbance_intensity'], 
                  alpha=0.5, s=20, c=df_disturbed['tumble_recovered'], cmap='RdYlGn')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Disturbance Intensity')
        ax.set_title('Disturbance Intensity Distribution (color=recovery)')
        ax.grid(True, alpha=0.3)
        
        # Add intensity range bands
        ax.axhspan(0.7, 0.9, alpha=0.1, color='green', label='EASY')
        ax.axhspan(0.9, 1.1, alpha=0.1, color='yellow', label='MEDIUM')
        ax.axhspan(1.1, 1.5, alpha=0.1, color='red', label='HARD')
        ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'training_learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    
    # Save as PDF for thesis
    pdf_path = output_dir / 'training_learning_curves.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {pdf_path}")
    
    plt.close()


def generate_statistics_table(df):
    """Generate statistics table for thesis"""
    print(f"\n📊 Generating statistics table...")
    
    # Overall statistics
    stats = {
        'total_episodes': len(df),
        'total_time_hours': df['elapsed_time_s'].iloc[-1] / 3600,
        'final_recovery_rate_last_50': df['rolling_50_recovery_rate'].iloc[-1],
        'final_reward_last_10': df['rolling_10_reward'].iloc[-1]
    }
    
    # Statistics by curriculum level
    for level in [0, 1, 2]:
        df_level = df[df['curriculum_level'] == level]
        if len(df_level) > 0:
            level_names = ['EASY', 'MEDIUM', 'HARD']
            stats[f'level_{level}_{level_names[level]}_episodes'] = len(df_level)
            
            # Calculate recovery rate for this level
            df_level_disturbed = df_level[df_level['tumble_initiated'] == 1]
            if len(df_level_disturbed) > 0:
                recovery_rate = df_level_disturbed['tumble_recovered'].mean() * 100
                stats[f'level_{level}_{level_names[level]}_recovery_rate'] = recovery_rate
                stats[f'level_{level}_{level_names[level]}_avg_intensity'] = df_level_disturbed['disturbance_intensity'].mean()
    
    # Find curriculum transitions
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['curriculum_level'] != df.iloc[i-1]['curriculum_level']:
            transitions.append({
                'episode': int(df.iloc[i]['episode']),
                'from_level': int(df.iloc[i-1]['curriculum_level']),
                'to_level': int(df.iloc[i]['curriculum_level']),
                'time_hours': df.iloc[i]['elapsed_time_s'] / 3600
            })
    
    # Print table
    print("\n" + "="*80)
    print("📊 TRAINING STATISTICS SUMMARY - Table 5.1")
    print("="*80)
    
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"  Total Episodes:           {stats['total_episodes']}")
    print(f"  Total Training Time:      {stats['total_time_hours']:.1f} hours")
    print(f"  Final Recovery Rate:      {stats['final_recovery_rate_last_50']:.1f}% (last 50)")
    print(f"  Final Avg Reward:         {stats['final_reward_last_10']:.1f} (last 10)")
    
    print(f"\n{'='*80}")
    print("CURRICULUM PROGRESSION - Table 5.2")
    print(f"{'='*80}")
    
    if transitions:
        print(f"\n{'Level Transition':<25} {'Episode':<15} {'Time (hours)':<15} {'Episodes Trained':<20}")
        print("-"*80)
        
        prev_episode = 1
        for t in transitions:
            level_names = ['EASY (0.7-0.9×)', 'MEDIUM (0.9-1.1×)', 'HARD (1.1-1.5×)']
            transition_str = f"Level {t['from_level']} → {t['to_level']}"
            episodes_trained = t['episode'] - prev_episode
            
            print(f"{transition_str:<25} {t['episode']:<15} {t['time_hours']:<15.1f} {episodes_trained:<20}")
            prev_episode = t['episode']
        
        # Final level
        final_episodes = stats['total_episodes'] - prev_episode + 1
        print(f"{'Level 2 (Final)':<25} {prev_episode:<15} {stats['total_time_hours']:<15.1f} {final_episodes:<20}")
    
    print(f"\n{'='*80}")
    print("BY CURRICULUM LEVEL - Table 5.4")
    print(f"{'='*80}")
    print(f"\n{'Level':<20} {'Episodes':<15} {'Recovery Rate':<20} {'Avg Intensity':<15}")
    print("-"*80)
    
    for level in [0, 1, 2]:
        level_names = ['EASY (0.7-0.9×)', 'MEDIUM (0.9-1.1×)', 'HARD (1.1-1.5×)']
        key_prefix = f'level_{level}_{["EASY", "MEDIUM", "HARD"][level]}'
        
        if f'{key_prefix}_episodes' in stats:
            episodes = stats[f'{key_prefix}_episodes']
            recovery = stats.get(f'{key_prefix}_recovery_rate', 0)
            intensity = stats.get(f'{key_prefix}_avg_intensity', 0)
            
            print(f"{level_names[level]:<20} {episodes:<15} {recovery:<20.1f} {intensity:<15.2f}")
    
    print("="*80 + "\n")
    
    # Generate LaTeX tables
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Table 5.1: Overall Training Statistics
    with open(output_dir / "table_5_1_training_statistics.tex", 'w') as f:
        f.write("% Table 5.1: Complete Training Progression\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Complete Training Progression Across Three Stages}\n")
        f.write("\\label{tab:training_progression}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Total Episodes & {stats['total_episodes']} \\\\\n")
        f.write(f"Total Training Time & {stats['total_time_hours']:.1f} hours \\\\\n")
        f.write(f"Final Recovery Rate (last 50) & {stats['final_recovery_rate_last_50']:.1f}\\% \\\\\n")
        f.write(f"Final Avg Reward (last 10) & {stats['final_reward_last_10']:.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Table 5.2: Curriculum Advancement
    with open(output_dir / "table_5_2_curriculum_advancement.tex", 'w') as f:
        f.write("% Table 5.2: Stage 3 Curriculum Advancement (Actual Results)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance-Gated Curriculum Advancement}\n")
        f.write("\\label{tab:curriculum_advancement}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Transition} & \\textbf{Episode} & \\textbf{Time (hours)} & \\textbf{Episodes Trained} \\\\\n")
        f.write("\\midrule\n")
        
        prev_episode = 1
        for t in transitions:
            episodes_trained = t['episode'] - prev_episode
            f.write(f"Level {t['from_level']} $\\rightarrow$ {t['to_level']} & {t['episode']} & {t['time_hours']:.1f} & {episodes_trained} \\\\\n")
            prev_episode = t['episode']
        
        final_episodes = stats['total_episodes'] - prev_episode + 1
        f.write(f"Level 2 (Final) & {prev_episode} & {stats['total_time_hours']:.1f} & {final_episodes} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"📄 LaTeX tables saved:")
    print(f"   - analysis_results/table_5_1_training_statistics.tex")
    print(f"   - analysis_results/table_5_2_curriculum_advancement.tex\n")
    
    return stats


def plot_curriculum_transitions(df, output_dir):
    """Plot curriculum level transitions with statistics"""
    print(f"\n📊 Generating curriculum transition plot...")
    
    output_dir = Path(output_dir)
    
    # Find transition points
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['curriculum_level'] != df.iloc[i-1]['curriculum_level']:
            transitions.append({
                'episode': df.iloc[i]['episode'],
                'from_level': df.iloc[i-1]['curriculum_level'],
                'to_level': df.iloc[i]['curriculum_level'],
                'time_hours': df.iloc[i]['elapsed_time_s'] / 3600
            })
    
    print(f"\n📈 Curriculum Transitions:")
    for t in transitions:
        level_names = ['EASY', 'MEDIUM', 'HARD']
        print(f"   Level {t['from_level']} ({level_names[int(t['from_level'])]}) → "
              f"Level {t['to_level']} ({level_names[int(t['to_level'])]})")
        print(f"      At episode {t['episode']} ({t['time_hours']:.1f}h)")
    
    # Create detailed plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot recovery rate with curriculum level background
    ax.plot(df['episode'], df['rolling_50_recovery_rate'], 
            linewidth=2.5, color='darkblue', label='Recovery Rate (Rolling 50)')
    
    # Add shaded regions for curriculum levels
    level_names = ['EASY\n(0.7-0.9×)', 'MEDIUM\n(0.9-1.1×)', 'HARD\n(1.1-1.5×)']
    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    
    prev_episode = 0
    for t in transitions:
        level = int(df.iloc[prev_episode]['curriculum_level'])
        ax.axvspan(prev_episode, t['episode'], alpha=0.2, color=colors[level], 
                   label=f"Level {level}: {level_names[level]}" if level < 2 else None)
        ax.axvline(x=t['episode'], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        prev_episode = t['episode']
    
    # Final level
    if len(df) > 0:
        final_level = int(df.iloc[-1]['curriculum_level'])
        ax.axvspan(prev_episode, len(df), alpha=0.2, color=colors[final_level],
                   label=f"Level {final_level}: {level_names[final_level]}")
    
    # Add threshold lines
    ax.axhline(y=80, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Level 0 Target (80%)')
    ax.axhline(y=70, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Level 1 Target (70%)')
    ax.axhline(y=60, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Level 2 Target (60%)')
    
    ax.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance-Gated Curriculum Progression with Recovery Rate', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'curriculum_progression.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    
    pdf_path = output_dir / 'curriculum_progression.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('--log', type=str, required=False,
                        help='Path to CSV log file')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Auto-find latest log if not specified
    if args.log is None:
        log_dir = Path('./logs/stage3/')
        if log_dir.exists():
            csv_files = list(log_dir.glob('*_episodes.csv'))
            if csv_files:
                args.log = str(sorted(csv_files)[-1])  # Get most recent
                print(f"📂 Auto-detected latest log: {args.log}")
            else:
                print("❌ No log files found in ./logs/stage3/")
                print("   Please specify --log path or run training first")
                return
        else:
            print("❌ ./logs/stage3/ directory not found")
            print("   Please run training with logging enabled first")
            return
    
    # Load data
    df = load_training_data(args.log)
    
    # Generate plots
    plot_learning_curves(df, args.output_dir)
    plot_curriculum_transitions(df, args.output_dir)
    
    # Generate statistics
    stats = generate_statistics_table(df)
    
    # Save statistics to JSON
    output_dir = Path(args.output_dir)
    stats_path = output_dir / 'training_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved to: {stats_path}")
    
    print(f"\n🎉 Analysis complete! Plots saved to: {args.output_dir}/")
    print(f"\n📊 Use these plots in your thesis:")
    print(f"   - training_learning_curves.pdf")
    print(f"   - curriculum_progression.pdf")


if __name__ == "__main__":
    main()