"""
Real-Time Training Monitor
==========================
Displays live training metrics in terminal during PPO training.
Updates every few seconds with latest statistics.
"""

import os
import time
import pandas as pd
from pathlib import Path
from typing import Optional
import sys


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    if abs(num) >= 1000:
        return f"{num:,.{decimals}f}"
    return f"{num:.{decimals}f}"


def create_progress_bar(percentage: float, width: int = 40) -> str:
    """Create ASCII progress bar."""
    filled = int(width * percentage)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage*100:.1f}%"


def monitor_training(
    log_dir: str = "./logs",
    refresh_interval: int = 5,
    total_timesteps: int = 200_000
):
    """
    Monitor training progress in real-time.
    
    Args:
        log_dir: Directory containing training logs
        refresh_interval: Seconds between updates
        total_timesteps: Total training timesteps for progress calculation
    """
    
    print("🚁 Drone Recovery Training Monitor")
    print("=" * 80)
    print(f"Log Directory: {log_dir}")
    print(f"Refresh Interval: {refresh_interval}s")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    # Find CSV log file
    log_path = Path(log_dir)
    
    last_update_time = 0
    iteration = 0
    
    try:
        while True:
            # Find most recent progress.csv
            csv_files = list(log_path.glob("**/progress.csv"))
            
            if not csv_files:
                print("⏳ Waiting for training to start...")
                print("   (Looking for progress.csv in log directory)")
                time.sleep(refresh_interval)
                continue
            
            # Use most recent CSV
            csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            
            # Check if file was updated
            current_mtime = csv_file.stat().st_mtime
            if current_mtime == last_update_time:
                time.sleep(1)
                continue
            
            last_update_time = current_mtime
            
            # Read CSV
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"⚠️  Error reading CSV: {e}")
                time.sleep(refresh_interval)
                continue
            
            if len(df) == 0:
                time.sleep(1)
                continue
            
            # Get latest metrics
            latest = df.iloc[-1]
            
            # Calculate moving averages
            window = min(10, len(df))
            if len(df) >= window:
                recent_df = df.tail(window)
                avg_return = recent_df.get('rollout/mean_return', pd.Series([0])).mean()
                avg_success = recent_df.get('rollout/success_rate', pd.Series([0])).mean()
            else:
                avg_return = latest.get('rollout/mean_return', 0)
                avg_success = latest.get('rollout/success_rate', 0)
            
            # Extract key metrics
            total_steps = int(latest.get('time/total_timesteps', 0))
            fps = latest.get('time/fps', 0)
            elapsed_time = latest.get('time/time_elapsed', 0)
            
            # Episode metrics
            ep_return = latest.get('episode/return', 0)
            ep_length = latest.get('episode/length', 0)
            
            # Loss metrics
            policy_loss = latest.get('train/policy_loss', 0)
            value_loss = latest.get('train/value_loss', 0)
            
            # Calculate progress
            progress = total_steps / total_timesteps if total_timesteps > 0 else 0
            progress = min(progress, 1.0)
            
            # Estimate time remaining
            if fps > 0:
                remaining_steps = max(0, total_timesteps - total_steps)
                eta_seconds = remaining_steps / fps
                eta_hours = eta_seconds / 3600
                eta_str = f"{eta_hours:.1f}h"
            else:
                eta_str = "N/A"
            
            # Clear screen and display
            clear_screen()
            iteration += 1
            
            print("=" * 80)
            print(f"🚁 DRONE RECOVERY TRAINING MONITOR (Update #{iteration})")
            print("=" * 80)
            print()
            
            # Progress bar
            print("📊 TRAINING PROGRESS")
            print(create_progress_bar(progress, width=60))
            print(f"   Steps: {format_number(total_steps, 0)} / {format_number(total_timesteps, 0)}")
            print(f"   ETA: {eta_str}")
            print()
            
            # Performance metrics
            print("⚡ PERFORMANCE")
            print(f"   Speed: {fps:.1f} steps/sec")
            print(f"   Elapsed: {elapsed_time/3600:.2f} hours")
            print()
            
            # Learning metrics
            print("📈 LEARNING METRICS (Last 10 Updates)")
            print(f"   Mean Return: {avg_return:>8.2f}")
            print(f"   Success Rate: {avg_success:>7.1%}")
            print()
            
            print("📋 LATEST EPISODE")
            print(f"   Return: {ep_return:>8.2f}")
            print(f"   Length: {ep_length:>8.0f} steps")
            print()
            
            # Loss metrics (if available)
            if not pd.isna(policy_loss):
                print("🎯 TRAINING LOSSES")
                print(f"   Policy Loss: {policy_loss:>8.4f}")
                print(f"   Value Loss: {value_loss:>8.4f}")
                print()
            
            # Learning status
            print("🔍 LEARNING STATUS")
            if avg_return > 40:
                status = "✅ EXCELLENT - Agent performing well"
            elif avg_return > 20:
                status = "🟢 GOOD - Learning progressing"
            elif avg_return > 0:
                status = "🟡 FAIR - Initial learning phase"
            elif avg_return > -50:
                status = "🟠 POOR - Struggling, may need tuning"
            else:
                status = "🔴 CRITICAL - Check hyperparameters"
            print(f"   {status}")
            print()
            
            print("=" * 80)
            print(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Next Refresh: {refresh_interval}s | Press Ctrl+C to stop")
            print("=" * 80)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_training_summary(log_dir: str = "./logs"):
    """Print summary of completed training."""
    
    log_path = Path(log_dir)
    csv_files = list(log_path.glob("**/progress.csv"))
    
    if not csv_files:
        print("❌ No training logs found")
        return
    
    csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("❌ Empty log file")
        return
    
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    
    total_steps = df['time/total_timesteps'].iloc[-1]
    total_time = df['time/time_elapsed'].iloc[-1] / 3600
    
    print(f"\n📈 Statistics:")
    print(f"   Total Steps: {total_steps:,.0f}")
    print(f"   Training Time: {total_time:.2f} hours")
    print(f"   Average Speed: {total_steps / (total_time * 3600):.1f} steps/sec")
    
    if 'rollout/mean_return' in df.columns:
        returns = df['rollout/mean_return'].dropna()
        print(f"\n🎯 Returns:")
        print(f"   Initial: {returns.iloc[0]:.2f}")
        print(f"   Final: {returns.iloc[-1]:.2f}")
        print(f"   Max: {returns.max():.2f}")
        print(f"   Improvement: {returns.iloc[-1] - returns.iloc[0]:+.2f}")
    
    if 'rollout/success_rate' in df.columns:
        success = df['rollout/success_rate'].dropna()
        print(f"\n✅ Success Rate:")
        print(f"   Initial: {success.iloc[0]:.1%}")
        print(f"   Final: {success.iloc[-1]:.1%}")
        print(f"   Max: {success.max():.1%}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor drone recovery training")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval (seconds)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total timesteps")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    
    args = parser.parse_args()
    
    if args.summary:
        print_training_summary(args.log_dir)
    else:
        monitor_training(
            log_dir=args.log_dir,
            refresh_interval=args.interval,
            total_timesteps=args.timesteps
        )