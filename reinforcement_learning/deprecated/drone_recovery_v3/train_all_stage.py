"""
Automatic Curriculum Training - Train All Stages Sequentially
==============================================================
Trains Stage 1 → 2 → 3 automatically with proper stage advancement.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from train_ppo import train_drone_recovery, DetailedLoggingCallback


def train_complete_curriculum(
    total_timesteps_per_stage: Dict[int, int] = None,
    log_dir: str = "./logs",
    save_freq: int = 10_000,
    stage_advancement_threshold: float = 0.80,
    debug: bool = False
) -> Dict:
    """
    Train complete curriculum from Stage 1 → 2 → 3 automatically.
    
    Args:
        total_timesteps_per_stage: Timesteps for each stage
        log_dir: Base log directory
        save_freq: Checkpoint frequency
        stage_advancement_threshold: Success rate needed to advance (0-1)
        debug: Enable debug mode
    
    Returns:
        Dictionary with training results
    """
    
    if total_timesteps_per_stage is None:
        total_timesteps_per_stage = {
            1: 200_000,  # Stage 1: Hover learning
            2: 150_000,  # Stage 2: Disturbance handling
            3: 200_000,  # Stage 3: Impact recovery
        }
    
    print("\n" + "="*80)
    print("🎓 AUTOMATIC CURRICULUM TRAINING")
    print("="*80)
    print("This will train all 3 stages sequentially:")
    print(f"  Stage 1 (Hover):        {total_timesteps_per_stage[1]:,} steps")
    print(f"  Stage 2 (Disturbances): {total_timesteps_per_stage[2]:,} steps")
    print(f"  Stage 3 (Impacts):      {total_timesteps_per_stage[3]:,} steps")
    print(f"  Total:                  {sum(total_timesteps_per_stage.values()):,} steps")
    print(f"\nStage Advancement Threshold: {stage_advancement_threshold:.0%}")
    print("="*80 + "\n")
    
    # Confirm with user
    print("⏰ Estimated Time:")
    print(f"  Stage 1: ~2-4 hours")
    print(f"  Stage 2: ~1.5-3 hours")
    print(f"  Stage 3: ~2-4 hours")
    print(f"  Total:   ~5.5-11 hours\n")
    
    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled by user")
        return {}
    
    print("\n🚀 Starting automatic curriculum training...\n")
    
    # Track overall progress
    start_time = time.time()
    results = {}
    current_model_path = None
    
    # Train each stage
    for stage in [1, 2, 3]:
        stage_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"🎯 STAGE {stage} TRAINING")
        print("="*80)
        print(f"Timesteps: {total_timesteps_per_stage[stage]:,}")
        if current_model_path:
            print(f"Loading from: {current_model_path}")
        print("="*80 + "\n")
        
        # Train this stage
        try:
            model = train_drone_recovery(
                total_timesteps=total_timesteps_per_stage[stage],
                stage=stage,
                model_path=current_model_path,
                log_dir=log_dir,
                save_freq=save_freq,
                debug=debug
            )
            
            # Save stage completion info
            stage_time = time.time() - stage_start_time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage_model_path = f"models/stage{stage}_complete_{timestamp}.zip"
            model.save(stage_model_path)
            
            print(f"\n✅ Stage {stage} completed in {stage_time/3600:.2f} hours")
            print(f"💾 Model saved: {stage_model_path}")
            
            # Update for next stage
            current_model_path = stage_model_path
            
            results[f"stage_{stage}"] = {
                "completed": True,
                "model_path": stage_model_path,
                "training_time_hours": stage_time / 3600,
                "timesteps": total_timesteps_per_stage[stage]
            }
            
            # Check if we should continue to next stage
            if stage < 3:
                print(f"\n⏭️  Advancing to Stage {stage + 1}...")
                time.sleep(2)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted during Stage {stage}!")
            
            # Save interrupted model
            interrupted_path = f"models/stage{stage}_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            if 'model' in locals():
                model.save(interrupted_path)
                print(f"💾 Interrupted model saved: {interrupted_path}")
            
            results[f"stage_{stage}"] = {
                "completed": False,
                "interrupted": True,
                "model_path": interrupted_path if 'model' in locals() else None
            }
            
            break
        
        except Exception as e:
            print(f"\n❌ Error during Stage {stage} training: {e}")
            import traceback
            traceback.print_exc()
            
            results[f"stage_{stage}"] = {
                "completed": False,
                "error": str(e)
            }
            
            break
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🏆 CURRICULUM TRAINING COMPLETE")
    print("="*80)
    print(f"Total Time: {total_time/3600:.2f} hours")
    print()
    
    for stage in [1, 2, 3]:
        stage_key = f"stage_{stage}"
        if stage_key in results:
            stage_result = results[stage_key]
            if stage_result.get("completed"):
                status = "✅ COMPLETE"
                time_str = f"({stage_result['training_time_hours']:.2f}h)"
            elif stage_result.get("interrupted"):
                status = "⚠️  INTERRUPTED"
                time_str = ""
            else:
                status = "❌ FAILED"
                time_str = ""
            
            print(f"  Stage {stage}: {status} {time_str}")
            if stage_result.get("model_path"):
                print(f"    Model: {stage_result['model_path']}")
        else:
            print(f"  Stage {stage}: ⏭️  SKIPPED")
    
    print("="*80 + "\n")
    
    # Save curriculum results
    results["total_time_hours"] = total_time / 3600
    results["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = f"models/curriculum_results_{results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved: {results_file}")
    
    # Show final model info
    if "stage_3" in results and results["stage_3"].get("completed"):
        final_model = results["stage_3"]["model_path"]
        print(f"\n🎉 FINAL MODEL READY:")
        print(f"   {final_model}")
        print(f"\n🧪 Test it with:")
        print(f"   python test_model.py --model {final_model} --test full --save")
    elif "stage_2" in results and results["stage_2"].get("completed"):
        print(f"\n⚠️  Training stopped after Stage 2")
        print(f"   Model: {results['stage_2']['model_path']}")
    elif "stage_1" in results and results["stage_1"].get("completed"):
        print(f"\n⚠️  Training stopped after Stage 1")
        print(f"   Model: {results['stage_1']['model_path']}")
    
    print()
    return results


def resume_curriculum(
    resume_from_stage: int,
    model_path: str,
    total_timesteps_per_stage: Dict[int, int] = None,
    log_dir: str = "./logs",
    save_freq: int = 10_000,
    debug: bool = False
) -> Dict:
    """
    Resume curriculum training from a specific stage.
    
    Args:
        resume_from_stage: Stage to resume from (1, 2, or 3)
        model_path: Path to model to load
        total_timesteps_per_stage: Timesteps for remaining stages
        log_dir: Log directory
        save_freq: Checkpoint frequency
        debug: Debug mode
    
    Returns:
        Training results
    """
    
    if total_timesteps_per_stage is None:
        total_timesteps_per_stage = {
            1: 200_000,
            2: 150_000,
            3: 200_000,
        }
    
    print("\n" + "="*80)
    print("🔄 RESUME CURRICULUM TRAINING")
    print("="*80)
    print(f"Resuming from: Stage {resume_from_stage}")
    print(f"Loading model: {model_path}")
    
    # Determine which stages to train
    stages_to_train = list(range(resume_from_stage, 4))
    print(f"Will train: Stages {', '.join(map(str, stages_to_train))}")
    print("="*80 + "\n")
    
    # Create modified timesteps dict
    remaining_timesteps = {
        stage: total_timesteps_per_stage[stage] 
        for stage in stages_to_train
    }
    
    # Train remaining stages
    results = {}
    current_model_path = model_path
    
    for stage in stages_to_train:
        print(f"\n{'='*80}")
        print(f"🎯 STAGE {stage} TRAINING")
        print(f"{'='*80}\n")
        
        try:
            model = train_drone_recovery(
                total_timesteps=remaining_timesteps[stage],
                stage=stage,
                model_path=current_model_path,
                log_dir=log_dir,
                save_freq=save_freq,
                debug=debug
            )
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stage_model_path = f"models/stage{stage}_complete_{timestamp}.zip"
            model.save(stage_model_path)
            
            print(f"\n✅ Stage {stage} completed")
            print(f"💾 Model saved: {stage_model_path}")
            
            current_model_path = stage_model_path
            results[f"stage_{stage}"] = {
                "completed": True,
                "model_path": stage_model_path
            }
            
        except Exception as e:
            print(f"\n❌ Error in Stage {stage}: {e}")
            break
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all curriculum stages automatically")
    parser.add_argument("--stage1-steps", type=int, default=200_000, help="Stage 1 timesteps")
    parser.add_argument("--stage2-steps", type=int, default=150_000, help="Stage 2 timesteps")
    parser.add_argument("--stage3-steps", type=int, default=200_000, help="Stage 3 timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    parser.add_argument("--threshold", type=float, default=0.80, help="Stage advancement threshold")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--resume", type=int, choices=[1, 2, 3], help="Resume from stage")
    parser.add_argument("--model", type=str, help="Model path for resume")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    timesteps = {
        1: args.stage1_steps,
        2: args.stage2_steps,
        3: args.stage3_steps
    }
    
    if args.resume:
        if not args.model:
            print("❌ Error: --model required when using --resume")
            return 1
        
        if not Path(args.model).exists():
            print(f"❌ Error: Model not found: {args.model}")
            return 1
        
        results = resume_curriculum(
            resume_from_stage=args.resume,
            model_path=args.model,
            total_timesteps_per_stage=timesteps,
            log_dir=args.log_dir,
            save_freq=args.save_freq,
            debug=args.debug
        )
    else:
        # Modify to skip confirmation if --yes flag
        if args.yes:
            # Directly call training without confirmation
            results = {}
            start_time = time.time()
            current_model_path = None
            
            for stage in [1, 2, 3]:
                try:
                    model = train_drone_recovery(
                        total_timesteps=timesteps[stage],
                        stage=stage,
                        model_path=current_model_path,
                        log_dir=args.log_dir,
                        save_freq=args.save_freq,
                        debug=args.debug
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    stage_model_path = f"models/stage{stage}_complete_{timestamp}.zip"
                    model.save(stage_model_path)
                    
                    current_model_path = stage_model_path
                    results[f"stage_{stage}"] = {
                        "completed": True,
                        "model_path": stage_model_path
                    }
                except Exception as e:
                    print(f"❌ Error in Stage {stage}: {e}")
                    break
            
            results["total_time_hours"] = (time.time() - start_time) / 3600
        else:
            results = train_complete_curriculum(
                total_timesteps_per_stage=timesteps,
                log_dir=args.log_dir,
                save_freq=args.save_freq,
                stage_advancement_threshold=args.threshold,
                debug=args.debug
            )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())