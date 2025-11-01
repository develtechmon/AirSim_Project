from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
from datetime import datetime
from drone_curriculum_env import DroneCurriculumRecoveryEnv

class CurriculumCallback(BaseCallback):
    """
    Automatically advance training stages based on performance
    """
    def __init__(self, env, stage_thresholds, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.env = env
        self.stage_thresholds = stage_thresholds
        self.current_stage = 1
        self.episode_rewards = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Collect episode statistics
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                
                # Check success (did it recover or stay stable?)
                if 'episode_stats' in info:
                    stats = info['episode_stats']
                    success = (stats['successful_recoveries'] > 0 or 
                             stats['crashes'] == 0)
                    self.episode_successes.append(float(success))
        
        # Check if we should advance stage
        if len(self.episode_rewards) >= 50:  # Check every 50 episodes
            mean_reward = np.mean(self.episode_rewards[-50:])
            success_rate = np.mean(self.episode_successes[-50:])
            
            # Check if we meet threshold for next stage
            if self.current_stage < 5:
                threshold = self.stage_thresholds.get(self.current_stage, {})
                
                if (mean_reward >= threshold.get('reward', 0) and
                    success_rate >= threshold.get('success_rate', 0)):
                    
                    self.current_stage += 1
                    
                    print(f"\n{'='*70}")
                    print(f"🎓 ADVANCING TO STAGE {self.current_stage}!")
                    print(f"   Mean Reward: {mean_reward:.2f} (threshold: {threshold.get('reward', 0)})")
                    print(f"   Success Rate: {success_rate*100:.1f}% (threshold: {threshold.get('success_rate', 0)*100:.1f}%)")
                    print(f"{'='*70}\n")
                    
                    # Update environment stage
                    if hasattr(self.env, 'env_method'):
                        self.env.env_method('set_training_stage', self.current_stage)
                    else:
                        self.env.set_training_stage(self.current_stage)
        
        return True

def make_env():
    def _init():
        env = DroneCurriculumRecoveryEnv()
        env = Monitor(env)
        return env
    return _init

def train_with_curriculum(total_timesteps=1000000):
    """
    Train with automatic curriculum progression
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./models/curriculum_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    print("\n" + "="*70)
    print("🎓 CURRICULUM LEARNING - DRONE RECOVERY")
    print("="*70)
    print("\nTraining Stages:")
    print("  Stage 1 (0-100k):    Basic hovering, NO disturbances")
    print("  Stage 2 (100k-300k): Light disturbances")
    print("  Stage 3 (300k-600k): Medium disturbances")
    print("  Stage 4 (600k-1M):   Heavy disturbances")
    print("  Stage 5 (1M+):       EXTREME scenarios")
    print("="*70)
    
    # Create environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Stage advancement thresholds
    stage_thresholds = {
        1: {'reward': 50.0, 'success_rate': 0.8},   # Must hover stably
        2: {'reward': 30.0, 'success_rate': 0.6},   # Handle light disturbances
        3: {'reward': 20.0, 'success_rate': 0.5},   # Handle medium disturbances
        4: {'reward': 10.0, 'success_rate': 0.4},   # Handle heavy disturbances
    }
    
    # Callbacks
    curriculum_callback = CurriculumCallback(env, stage_thresholds, verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_curriculum"
    )
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # Standard LR
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        device="cuda",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )
    
    print(f"\n🚀 Starting curriculum training for {total_timesteps:,} steps...")
    print(f"💾 Saving to: {save_dir}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted")
    
    # Save final model
    model.save(os.path.join(save_dir, "ppo_curriculum_final"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete!")
    print(f"   Final stage reached: {curriculum_callback.current_stage}")
    print(f"   Saved to: {save_dir}")
    
    return model, env, save_dir

if __name__ == "__main__":
    print("\n🎓 CURRICULUM LEARNING TRAINING")
    print("This will automatically progress through difficulty stages")
    print("as the drone learns. Much better than jumping straight to")
    print("violent flips!\n")
    
    timesteps = input("Total timesteps (default 1000000): ").strip()
    timesteps = int(timesteps) if timesteps else 1000000
    
    input("Press ENTER to start training...")
    
    train_with_curriculum(total_timesteps=timesteps)