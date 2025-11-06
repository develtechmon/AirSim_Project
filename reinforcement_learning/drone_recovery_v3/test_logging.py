"""
Quick Logging Test
==================
Tests that logging works before full training.
Runs for just 5000 steps to verify data appears in CSV and TensorBoard.
"""

import os
import sys
from pathlib import Path

def test_logging():
    """Quick test that logging works."""
    
    print("="*80)
    print("🧪 QUICK LOGGING TEST")
    print("="*80)
    print("This will train for 5000 steps (~2-5 minutes) to verify logging works")
    print("="*80 + "\n")
    
    # Import here to show any import errors
    try:
        from train_drone_recovery import train_drone_recovery
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Run short training
    try:
        print("🚀 Starting short training run...\n")
        model = train_drone_recovery(
            total_timesteps=5000,  # Just 5k steps
            stage=1,
            log_dir="./logs_test",
            save_freq=2500,
            debug=False
        )
        
        print("\n" + "="*80)
        print("✅ TEST COMPLETED")
        print("="*80)
        
        # Check if files were created
        log_dir = Path("./logs_test")
        
        print("\n🔍 Checking for generated files...\n")
        
        # Find the timestamped directory
        subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("❌ No log subdirectory found!")
            return False
        
        latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
        print(f"📁 Log directory: {latest_dir}\n")
        
        # Check for CSV
        csv_file = latest_dir / "progress.csv"
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"✅ progress.csv created with {len(df)} rows")
            if len(df) > 0:
                print(f"   Columns: {list(df.columns)[:5]}...")
        else:
            print(f"❌ progress.csv not found at {csv_file}")
            return False
        
        # Check for monitor.csv
        monitor_file = latest_dir / "monitor.csv"
        if monitor_file.exists():
            with open(monitor_file, 'r') as f:
                lines = f.readlines()
            print(f"✅ monitor.csv created with {len(lines)-1} episode records")
        else:
            print(f"⚠️  monitor.csv not found at {monitor_file}")
        
        # Check for TensorBoard files
        tb_files = list(latest_dir.glob("events.out.tfevents.*"))
        if tb_files:
            print(f"✅ TensorBoard event files created ({len(tb_files)} files)")
        else:
            print("⚠️  No TensorBoard files found")
        
        print("\n" + "="*80)
        print("📊 TO VIEW YOUR DATA:")
        print("="*80)
        print(f"\n1. TensorBoard:")
        print(f"   tensorboard --logdir={latest_dir}")
        print(f"   Then open: http://localhost:6006")
        
        print(f"\n2. CSV:")
        print(f"   Open: {csv_file}")
        
        print(f"\n3. Monitor:")
        print(f"   python monitor_training.py --log-dir {latest_dir}")
        
        print("\n" + "="*80)
        print("✅ LOGGING TEST PASSED - Ready for full training!")
        print("="*80)
        print("\nRun full training with:")
        print("   python train_drone_recovery.py --stage 1 --timesteps 200000\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_logging()
    sys.exit(0 if success else 1)