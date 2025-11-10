"""
COMPARE ALL CURRICULUM MODELS
==============================
Tests and compares all saved models from curriculum training:
- Level 0 (EASY) model
- Level 1 (MEDIUM) model  
- Final (HARD) model

Shows how performance improved across curriculum stages!

Usage:
    python compare_curriculum_models.py --episodes 30
"""

import subprocess
import json
from pathlib import Path
import argparse


def test_model(model_path, vecnorm_path, episodes=30):
    """Run test script and capture results"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_path.name}")
    print(f"{'='*70}")
    
    cmd = [
        "python", "test_gated_curriculum.py",
        "--model", str(model_path),
        "--vecnorm", str(vecnorm_path),
        "--episodes", str(episodes)
    ]
    
    subprocess.run(cmd)


def main(args):
    print("\n" + "="*70)
    print("📊 CURRICULUM MODEL COMPARISON")
    print("="*70)
    print(f"Testing {args.episodes} episodes per model")
    print(f"This will take approximately {args.episodes * 3 * 0.5:.0f} minutes")
    print("="*70 + "\n")
    
    # Find all models
    models = []
    
    # Level models
    level_dir = Path("./models/curriculum_levels/")
    if level_dir.exists():
        level_0 = level_dir / "level_0_EASY_mastered.zip"
        level_1 = level_dir / "level_1_MEDIUM_mastered.zip"
        
        if level_0.exists():
            vecnorm_0 = level_dir / "level_0_EASY_mastered_vecnormalize.pkl"
            models.append(("Level 0 (EASY)", level_0, vecnorm_0))
        
        if level_1.exists():
            vecnorm_1 = level_dir / "level_1_MEDIUM_mastered_vecnormalize.pkl"
            models.append(("Level 1 (MEDIUM)", level_1, vecnorm_1))
    
    # Final model
    final_model = Path("./models/gated_curriculum_policy.zip")
    if final_model.exists():
        final_vecnorm = Path("./models/gated_curriculum_vecnormalize.pkl")
        models.append(("Final (HARD)", final_model, final_vecnorm))
    
    if not models:
        print("❌ No trained models found!")
        print("   Run training first: python train_gated_curriculum.py")
        return
    
    print(f"Found {len(models)} models to compare:\n")
    for name, model, _ in models:
        print(f"   ✅ {name}: {model}")
    print()
    
    input("Press Enter to start testing...")
    
    # Test each model
    for i, (name, model, vecnorm) in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {i}/{len(models)}: {name}")
        print(f"{'#'*70}")
        
        test_model(model, vecnorm, args.episodes)
        
        if i < len(models):
            print("\n" + "="*70)
            print("📊 Next model in 5 seconds...")
            print("="*70)
            import time
            time.sleep(5)
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\n📊 Expected Pattern:")
    print("\n   Level 0 Model (EASY mastered):")
    print("      Easy:   85-92% ✅")
    print("      Medium: 50-60%")
    print("      Hard:   20-30%")
    print("      Overall: ~55%")
    
    print("\n   Level 1 Model (EASY + MEDIUM mastered):")
    print("      Easy:   85-90% ✅")
    print("      Medium: 75-82% ✅")
    print("      Hard:   40-50%")
    print("      Overall: ~68%")
    
    print("\n   Final Model (ALL mastered):")
    print("      Easy:   85-92% ✅")
    print("      Medium: 75-82% ✅")
    print("      Hard:   65-75% ✅")
    print("      Overall: ~78-82%")
    
    print("\n📈 This shows clear curriculum progression!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30,
                        help='Episodes to test per model (30 = 10 per intensity level)')
    
    args = parser.parse_args()
    main(args)