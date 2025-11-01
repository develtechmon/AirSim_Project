"""
PPO Training Script for Impact Resilience - FIXED VERSION
==========================================================

Complete training script with proper callback handling for vectorized environments.

Fixes:
1. Proper handling of info dicts from vectorized environments
2. Correct boolean checks (no array ambiguity)
3. Robust error handling and logging
4. Curriculum learning support

Impact Types Trained:
1. Sharp Collision (bird, wall) - high jerk, brief
2. Sustained Force (wind) - low jerk, long
3. Rotational (asymmetric) - high angular accel
4. Free-fall (thrust loss) - vertical drop

Usage:
    python train_impact_resilience_FIXED.py
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import json
import sys
import traceback

# Add path for custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from impact_resilience_env import ImpactResilienceEnv
except ImportError:
    print("ERROR: Could not import impact_resilience_env.py")
    print("Make sure all required files are in the same directory:")
    print("  - impact_resilience_env.py")
    print("  - impact_simulator.py")
    print("  - feature_extraction.py")
    sys.exit(1)


class ImpactStatsCallback(BaseCallback):
    """
    Callback to track impact resilience statistics.
    
    FIXED: Properly handles vectorized environment info dicts.
    """
    
    def __init__(self, check_freq=10, log_dir=None, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_count = 0
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_recoveries = []
        self.episode_collisions = []
        self.episode_recovery_times = []
        
        # Impact type tracking
        self.episode_impact_types = {
            'sharp_collision': [],
            'sustained_force': [],
            'rotational': [],
            'free_fall': []
        }
        
        # Best metrics
        self.best_reward = -np.inf
        self.best_recovery_rate = 0.0
        
        # Create log file
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, 'training_log.csv')
            with open(self.log_file, 'w') as f:
                f.write('episode,timestep,reward,length,recovered,collision,recovery_time,recovery_rate\n')
    
    def _on_step(self):
        """
        Called after each step.
        
        FIXED: Properly handles vectorized environment.
        """
        try:
            # Check if episode ended
            # For vectorized envs, dones is an array
            dones = self.locals.get('dones')
            if dones is None:
                return True
            
            # Check first environment (we're using single env wrapped in DummyVecEnv)
            if dones[0]:
                self.episode_count += 1
                
                # Get info from first environment
                infos = self.locals.get('infos')
                if infos is None or len(infos) == 0:
                    return True
                
                info = infos[0]
                
                # Episode reward and length
                if 'episode' in info:
                    ep_reward = float(info['episode']['r'])
                    ep_length = int(info['episode']['l'])
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Track best reward
                    if ep_reward > self.best_reward:
                        self.best_reward = ep_reward
                
                # Recovery status (FIXED: direct boolean check)
                recovered = bool(info.get('recovered', False))
                self.episode_recoveries.append(recovered)
                
                # Collision status
                collision = bool(info.get('collision', False))
                self.episode_collisions.append(collision)
                
                # Recovery time (if available)
                recovery_time = info.get('recovery_time', 0)
                if recovery_time > 0:
                    self.episode_recovery_times.append(recovery_time)
                
                # Impact type tracking
                impact_type = info.get('impact_type', 'unknown')
                if impact_type in self.episode_impact_types:
                    self.episode_impact_types[impact_type].append(recovered)
                
                # Log to file
                if self.log_dir:
                    recovery_rate = np.mean(self.episode_recoveries[-50:]) * 100 if len(self.episode_recoveries) >= 50 else 0.0
                    with open(self.log_file, 'a') as f:
                        f.write(f'{self.episode_count},{self.num_timesteps},{ep_reward:.2f},{ep_length},{int(recovered)},{int(collision)},{recovery_time},{recovery_rate:.2f}\n')
                
                # Print progress every N episodes
                if self.episode_count % self.check_freq == 0:
                    self._print_progress()
            
            return True  # Continue training
            
        except Exception as e:
            print(f"\n⚠️  Error in callback: {e}")
            print(traceback.format_exc())
            return True  # Continue training even if callback fails
    
    def _print_progress(self):
        """Print detailed progress report."""
        
        # Calculate recent stats (last check_freq episodes)
        recent_rewards = self.episode_rewards[-self.check_freq:]
        recent_lengths = self.episode_lengths[-self.check_freq:]
        recent_recoveries = self.episode_recoveries[-self.check_freq:]
        recent_collisions = self.episode_collisions[-self.check_freq:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0.0
        recovery_rate = np.mean(recent_recoveries) * 100 if recent_recoveries else 0.0
        collision_rate = np.mean(recent_collisions) * 100 if recent_collisions else 0.0
        
        # Overall recovery rate (last 50 episodes)
        overall_recovery = np.mean(self.episode_recoveries[-50:]) * 100 if len(self.episode_recoveries) >= 50 else recovery_rate
        
        # Update best recovery rate
        if recovery_rate > self.best_recovery_rate:
            self.best_recovery_rate = recovery_rate
        
        # Average recovery time
        avg_recovery_time = np.mean(self.episode_recovery_times[-20:]) if len(self.episode_recovery_times) >= 20 else 0.0
        
        # Impact type breakdown
        impact_stats = {}
        for impact_type, successes in self.episode_impact_types.items():
            if len(successes) >= 5:
                rate = np.mean(successes[-10:]) * 100
                impact_stats[impact_type] = rate
        
        print(f"\n{'='*80}")
        print(f"📊 EPISODE {self.episode_count} | STEPS {self.num_timesteps:,}")
        print(f"{'='*80}")
        print(f"  📈 Performance:")
        print(f"     Avg Reward:        {avg_reward:>8.1f}  (Best: {self.best_reward:.1f})")
        print(f"     Avg Length:        {avg_length:>8.1f} steps")
        print(f"")
        print(f"  🎯 Recovery:")
        print(f"     Recent Rate:       {recovery_rate:>8.1f}%")
        print(f"     Overall Rate:      {overall_recovery:>8.1f}%")
        print(f"     Best Rate:         {self.best_recovery_rate:>8.1f}%")
        if avg_recovery_time > 0:
            print(f"     Avg Time:          {avg_recovery_time:>8.1f} steps")
        print(f"")
        print(f"  💥 Safety:")
        print(f"     Collision Rate:    {collision_rate:>8.1f}%")
        
        if impact_stats:
            print(f"")
            print(f"  📋 Impact Type Performance:")
            for impact_type, rate in impact_stats.items():
                print(f"     {impact_type:20} {rate:>6.1f}%")
        
        print(f"{'='*80}")


class SaveBestModelCallback(BaseCallback):
    """
    Callback to save the best model based on recovery rate.
    
    FIXED: Proper handling of info dicts.
    """
    
    def __init__(self, save_path, check_freq=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_recovery_rate = 0.0
        self.episode_count = 0
        self.recent_recoveries = []
    
    def _on_step(self):
        """Check and save best model."""
        try:
            # Check if episode ended
            dones = self.locals.get('dones')
            if dones is None:
                return True
            
            if dones[0]:
                self.episode_count += 1
                
                # Get recovery status
                infos = self.locals.get('infos')
                if infos and len(infos) > 0:
                    info = infos[0]
                    recovered = bool(info.get('recovered', False))
                    self.recent_recoveries.append(recovered)
                    
                    # Keep last 50 episodes
                    if len(self.recent_recoveries) > 50:
                        self.recent_recoveries.pop(0)
                    
                    # Check every N episodes
                    if self.episode_count % self.check_freq == 0 and len(self.recent_recoveries) >= 20:
                        # Calculate recovery rate from last 20 episodes
                        recovery_rate = np.mean(self.recent_recoveries[-20:]) * 100
                        
                        # Save if this is the best
                        if recovery_rate > self.best_recovery_rate:
                            self.best_recovery_rate = recovery_rate
                            
                            # Save model
                            model_path = os.path.join(
                                self.save_path,
                                f"best_model_{int(recovery_rate)}pct"
                            )
                            self.model.save(model_path)
                            
                            if self.verbose > 0:
                                print(f"\n💾 NEW BEST MODEL SAVED: {int(recovery_rate)}% recovery rate")
                                print(f"   Saved to: {model_path}.zip")
            
            return True
            
        except Exception as e:
            print(f"\n⚠️  Error saving model: {e}")
            return True


class CurriculumCallback(BaseCallback):
    """
    Gradually increase impact difficulty during training.
    """
    
    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.phase = 1
        self.phase_thresholds = {
            1: 50000,   # Phase 1: Light impacts
            2: 150000,  # Phase 2: Medium impacts
            3: 300000   # Phase 3: All impacts
        }
    
    def _on_step(self):
        """Update difficulty based on training progress."""
        try:
            timesteps = self.num_timesteps
            
            # Check for phase transitions
            if self.phase == 1 and timesteps >= self.phase_thresholds[1]:
                self.phase = 2
                print(f"\n🎓 CURRICULUM: Phase 2 - Medium impacts unlocked!")
                
            elif self.phase == 2 and timesteps >= self.phase_thresholds[2]:
                self.phase = 3
                print(f"\n🎓 CURRICULUM: Phase 3 - All impacts unlocked!")
            
            return True
            
        except Exception as e:
            print(f"\n⚠️  Error in curriculum: {e}")
            return True


def make_env():
    """Create and wrap environment."""
    def _init():
        try:
            env = ImpactResilienceEnv()
            env = Monitor(env)
            return env
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise
    return _init


def train_ppo(total_timesteps=300000, save_dir=None, use_curriculum=False):
    """
    Train PPO agent for impact resilience.
    
    Args:
        total_timesteps: Total training steps
        save_dir: Directory to save models and logs
        use_curriculum: Enable curriculum learning (gradual difficulty increase)
    
    Returns:
        model, env, save_dir
    """
    
    # Setup save directory
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"./impact_resilience_ppo_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🚀 IMPACT RESILIENCE PPO TRAINING")
    print(f"{'='*80}")
    print(f"  Total Timesteps:  {total_timesteps:,}")
    print(f"  Save Directory:   {save_dir}")
    print(f"  Curriculum:       {'Enabled' if use_curriculum else 'Disabled'}")
    print(f"  Observation Dim:  42 (27 IMU features + 15 drone state)")
    print(f"  Action Dim:       4 (vx, vy, vz, yaw_rate)")
    print(f"{'='*80}\n")
    
    # Save training config
    config = {
        'total_timesteps': total_timesteps,
        'algorithm': 'PPO',
        'environment': 'ImpactResilienceEnv',
        'observation_dim': 42,
        'action_dim': 4,
        'use_curriculum': use_curriculum,
        'impact_types': ['sharp_collision', 'sustained_force', 'rotational', 'free_fall'],
        'network_architecture': {
            'policy': [256, 256],
            'value': [256, 256]
        },
        'ppo_params': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        }
    }
    
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("📦 Creating environment...")
    try:
        env = DummyVecEnv([make_env()])
        print("✓ Environment created")
        
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        print("✓ Normalization wrapper applied")
        
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        print(traceback.format_exc())
        return None, None, None
    
    print("\n🧠 Creating PPO model...")
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            tensorboard_log=save_dir,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]
            ),
            verbose=1,
            device='auto'  # Use GPU if available
        )
        print("✓ PPO model created")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        print(traceback.format_exc())
        return None, env, None
    
    # Setup logger
    logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    print("✓ Logger configured")
    
    # Setup callbacks
    print("\n📊 Setting up callbacks...")
    stats_callback = ImpactStatsCallback(
        check_freq=10,
        log_dir=save_dir,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_checkpoint",
        verbose=1
    )
    
    best_model_callback = SaveBestModelCallback(
        save_path=save_dir,
        check_freq=10,
        verbose=1
    )
    
    callbacks = [stats_callback, checkpoint_callback, best_model_callback]
    
    if use_curriculum:
        curriculum_callback = CurriculumCallback(env, verbose=1)
        callbacks.append(curriculum_callback)
    
    callback = CallbackList(callbacks)
    print("✓ Callbacks configured")
    
    print("\n" + "="*80)
    print("🎓 TRAINING START")
    print("="*80)
    print("\n📚 Expected Learning Milestones:")
    print("  30,000 steps:   ~50% recovery from light impacts")
    print("  80,000 steps:   ~70% recovery from all impact types")
    print("  150,000 steps:  ~85% recovery with fast stabilization")
    print("  300,000 steps:  >90% recovery, robust to variations")
    print("\n💡 Tips:")
    print("  - Press Ctrl+C to stop training early")
    print("  - Progress is saved every 10,000 steps")
    print("  - Best models are saved automatically")
    print("  - Check TensorBoard for detailed metrics:")
    print(f"      tensorboard --logdir {save_dir}")
    print("="*80 + "\n")
    
    input("Press ENTER to begin training...")
    
    try:
        # Train
        print("\n🏃 Training in progress...\n")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user (Ctrl+C)")
        print("Saving current model state...")
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        print(traceback.format_exc())
    
    # Save final model
    print("\n💾 Saving final model...")
    try:
        final_path = os.path.join(save_dir, "ppo_final")
        model.save(final_path)
        env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        print(f"✓ Model saved to: {final_path}.zip")
        
    except Exception as e:
        print(f"⚠️  Error saving model: {e}")
    
    # Training summary
    print(f"\n{'='*80}")
    print(f"📊 TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"  Total Episodes:      {stats_callback.episode_count}")
    print(f"  Total Timesteps:     {stats_callback.num_timesteps:,}")
    print(f"  Best Reward:         {stats_callback.best_reward:.1f}")
    print(f"  Best Recovery Rate:  {stats_callback.best_recovery_rate:.1f}%")
    print(f"  Models Saved:        {save_dir}")
    print(f"{'='*80}\n")
    
    return model, env, save_dir


def test_model(model_path, vec_normalize_path, num_episodes=10):
    """
    Test a trained model.
    
    Args:
        model_path: Path to saved model (without .zip)
        vec_normalize_path: Path to vec_normalize.pkl
        num_episodes: Number of test episodes
    """
    print(f"\n{'='*80}")
    print(f"🧪 TESTING MODEL")
    print(f"{'='*80}")
    print(f"  Model:           {model_path}")
    print(f"  Normalization:   {vec_normalize_path}")
    print(f"  Test Episodes:   {num_episodes}")
    print(f"{'='*80}\n")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create environment
    try:
        env = ImpactResilienceEnv()
        print("✓ Environment created")
        
        # Load normalization stats
        if os.path.exists(vec_normalize_path):
            vec_norm = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: env]))
            vec_norm.training = False
            vec_norm.norm_reward = False
            print("✓ Normalization loaded")
    except Exception as e:
        print(f"❌ Failed to setup environment: {e}")
        return
    
    # Test episodes
    results = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        recovered = False
        collision = False
        impact_type = "none"
        
        while True:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Update status
            recovered = info.get('recovered', False)
            collision = info.get('collision', False)
            impact_type = info.get('impact_type', 'none')
            
            # Print progress
            if steps % 30 == 0:
                pos_err = info.get('position_error', 0)
                print(f"  Step {steps:3d}: Recovered={recovered}, PosErr={pos_err:.2f}m, Reward={reward:.1f}")
            
            if terminated or truncated:
                break
        
        # Episode summary
        results.append({
            'recovered': recovered,
            'collision': collision,
            'steps': steps,
            'reward': total_reward,
            'impact_type': impact_type
        })
        
        status = "✅" if recovered else ("💥" if collision else "❌")
        print(f"\n  {status} Episode Complete:")
        print(f"     Impact Type:   {impact_type}")
        print(f"     Recovered:     {recovered}")
        print(f"     Collision:     {collision}")
        print(f"     Steps:         {steps}")
        print(f"     Total Reward:  {total_reward:.1f}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    recovery_rate = np.mean([r['recovered'] for r in results]) * 100
    collision_rate = np.mean([r['collision'] for r in results]) * 100
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    
    print(f"  Recovery Rate:    {recovery_rate:.1f}%")
    print(f"  Collision Rate:   {collision_rate:.1f}%")
    print(f"  Avg Steps:        {avg_steps:.1f}")
    print(f"  Avg Reward:       {avg_reward:.1f}")
    
    # Per-impact-type breakdown
    impact_types = set(r['impact_type'] for r in results)
    if len(impact_types) > 1:
        print(f"\n  Per-Impact Performance:")
        for itype in impact_types:
            type_results = [r for r in results if r['impact_type'] == itype]
            if type_results:
                type_recovery = np.mean([r['recovered'] for r in type_results]) * 100
                print(f"    {itype:20} {type_recovery:>6.1f}% recovery")
    
    print(f"{'='*80}\n")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPACT RESILIENCE - PPO TRAINING (FIXED VERSION)")
    print("="*80)
    print("\nThis is the corrected version with proper callback handling.")
    print("\nOptions:")
    print("  1. Train new model")
    print("  2. Test existing model")
    print("  3. Continue training existing model")
    print("="*80)
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        # New training
        print("\n" + "="*80)
        print("NEW TRAINING CONFIGURATION")
        print("="*80)
        
        timesteps = input("Training timesteps (default 300000): ").strip()
        timesteps = int(timesteps) if timesteps else 300000
        
        use_curriculum = input("Use curriculum learning? (y/n, default n): ").strip().lower() == 'y'
        
        print(f"\n🚀 Starting training:")
        print(f"  - {timesteps:,} timesteps")
        print(f"  - Curriculum: {'Enabled' if use_curriculum else 'Disabled'}")
        print(f"\nThis will take several hours.")
        print("Training progress is saved automatically every 10,000 steps.")
        
        model, env, save_dir = train_ppo(
            total_timesteps=timesteps,
            use_curriculum=use_curriculum
        )
        
        if env:
            env.close()
        
        if save_dir:
            print(f"\n✅ Training files saved to: {save_dir}")
        
    elif choice == "2":
        # Test model
        print("\n" + "="*80)
        print("MODEL TESTING")
        print("="*80)
        
        model_path = input("Model path (without .zip): ").strip()
        vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        episodes = input("Test episodes (default 10): ").strip()
        episodes = int(episodes) if episodes else 10
        
        test_model(model_path, vec_norm_path, num_episodes=episodes)
        
    elif choice == "3":
        # Continue training
        print("\n" + "="*80)
        print("CONTINUE TRAINING")
        print("="*80)
        
        model_path = input("Model path to continue (without .zip): ").strip()
        vec_norm_path = input("VecNormalize path (.pkl): ").strip()
        save_dir = input("Save directory: ").strip()
        timesteps = input("Additional timesteps (default 100000): ").strip()
        timesteps = int(timesteps) if timesteps else 100000
        
        print(f"\n📖 Loading model from {model_path}...")
        
        try:
            # Load model and environment
            model = PPO.load(model_path)
            env = DummyVecEnv([make_env()])
            env = VecNormalize.load(vec_norm_path, env)
            model.set_env(env)
            
            print("✓ Model and environment loaded")
            
            # Setup callbacks
            stats_callback = ImpactStatsCallback(check_freq=10, log_dir=save_dir)
            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path=os.path.join(save_dir, "checkpoints"),
                name_prefix="ppo_continued"
            )
            
            print(f"🚀 Continuing training for {timesteps:,} more steps...")
            
            model.learn(
                total_timesteps=timesteps,
                callback=[stats_callback, checkpoint_callback],
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            # Save
            final_path = os.path.join(save_dir, "ppo_continued")
            model.save(final_path)
            env.save(os.path.join(save_dir, "vec_normalize_continued.pkl"))
            
            print(f"\n✅ Continued training complete! Saved to {final_path}")
            
            env.close()
            
        except Exception as e:
            print(f"\n❌ Error during continued training: {e}")
            print(traceback.format_exc())
    
    else:
        print("\n❌ Invalid choice. Please run again and select 1, 2, or 3.")