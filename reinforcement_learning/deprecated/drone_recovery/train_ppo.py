import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from airsim_env import AirSimDroneEnv
from curriculum_manager import CurriculumManager
import gymnasium as gym
import os
import time

class CustomNetwork(BaseFeaturesExtractor):
    """
    Custom feature extractor matching imitation learning architecture.
    13 → 256 → 256
    """
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=256)
        
        n_input = observation_space.shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

class DetailedProgressCallback(BaseCallback):
    """
    Enhanced callback with detailed progress tracking.
    
    CHANGES:
    1. Fixed episode length tracking (was showing 0)
    2. Added proper termination reason detection
    3. Improved ETA calculation
    """
    
    def __init__(self, curriculum_manager, env, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.env = env
        
        # Episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0  # ADDED: Track current episode length
        self.episode_count = 0
        self.episode_lengths = []
        self.episode_rewards = []
        
        # Step tracking
        self.total_timesteps = total_timesteps
        self.current_timestep = 0
        
        # Success tracking
        self.success_count = 0
        self.crash_count = 0
        self.flip_count = 0
        self.timeout_count = 0
        
        # Timing
        self.start_time = time.time()
        self.last_print_time = time.time()
        
        # Stage tracking
        self.episodes_in_stage = 0
        
    def _on_step(self):
        self.current_timestep += 1
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1  # ADDED: Increment episode length
        
        # Check for episode end
        done = self.locals.get('dones', [False])[0]
        
        if done:
            self.episode_count += 1
            self.episodes_in_stage += 1
            
            # Store episode stats
            self.episode_lengths.append(self.current_episode_length)
            self.episode_rewards.append(self.current_episode_reward)
            
            # FIXED: Get termination reason from observations
            obs = self.locals['new_obs'][0]
            
            # Determine why episode ended
            if obs[2] > -0.5:  # Crashed (too close to ground)
                self.crash_count += 1
                termination = "💥 CRASH"
            elif obs[3] < 0.3:  # Flipped (qw too low)
                self.flip_count += 1
                termination = "🔄 FLIP"
            elif self.current_episode_length >= 500:  # Max steps
                self.success_count += 1
                termination = "✅ SUCCESS"
            elif np.linalg.norm(obs[0:2]) > 20:  # Out of bounds
                self.crash_count += 1
                termination = "🚫 OUT OF BOUNDS"
            else:
                self.timeout_count += 1
                termination = "⚠️  OTHER"
            
            # Record performance for curriculum
            stage = self.curriculum.get_current_stage()
            self.curriculum.record_episode(stage, self.current_episode_reward)
            
            # Calculate statistics
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            
            recent_lengths = self.episode_lengths[-10:] if len(self.episode_lengths) >= 10 else self.episode_lengths
            avg_length = np.mean(recent_lengths)
            
            # Calculate progress
            progress = (self.current_timestep / self.total_timesteps) * 100
            
            # Calculate ETA
            elapsed_time = time.time() - self.start_time
            steps_per_sec = self.current_timestep / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = self.total_timesteps - self.current_timestep
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            
            # Success rate
            total_episodes = self.success_count + self.crash_count + self.flip_count + self.timeout_count
            success_rate = (self.success_count / total_episodes * 100) if total_episodes > 0 else 0
            
            # Print detailed progress
            print(f"\n{'='*80}")
            print(f"📊 EPISODE {self.episode_count} | STAGE {stage} | STEP {self.current_timestep:,}/{self.total_timesteps:,} ({progress:.1f}%)")
            print(f"{'='*80}")
            print(f"  Reward:          {self.current_episode_reward:>8.2f} | Avg(10): {avg_reward:>8.2f}")
            print(f"  Episode Length:  {self.current_episode_length:>8} steps | Avg(10): {avg_length:>8.1f}")
            print(f"  Termination:     {termination}")
            print(f"  Success Rate:    {success_rate:>7.1f}% ({self.success_count}/{total_episodes})")
            print(f"  Crashes:         {self.crash_count:>8} | Flips: {self.flip_count:>8}")
            print(f"  Steps/sec:       {steps_per_sec:>8.1f} | ETA: {eta_minutes:.1f} min")
            print(f"{'='*80}")
            
            # Check for stage advancement
            if self.curriculum.should_advance():
                new_stage = self.curriculum.advance_stage()
                self.env.set_stage(new_stage)
                
                print(f"\n{'🎉'*40}")
                print(f"🚀 CURRICULUM ADVANCED TO STAGE {new_stage}!")
                print(f"📈 Stage {stage} completed in {self.episodes_in_stage} episodes")
                print(f"{'🎉'*40}\n")
                
                self.model.save(f"data/ppo_stage{new_stage-1}_checkpoint")
                
                # Reset stage tracking
                self.episodes_in_stage = 0
                self.success_count = 0
                self.crash_count = 0
                self.flip_count = 0
                self.timeout_count = 0
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0  # ADDED: Reset episode length
        
        # Print rollout summary every 2048 steps
        current_time = time.time()
        if self.current_timestep % 2048 == 0:
            elapsed = current_time - self.last_print_time
            fps = 2048 / elapsed if elapsed > 0 else 0
            
            print(f"\n{'─'*80}")
            print(f"🔄 ROLLOUT COMPLETE | Step {self.current_timestep:,} | FPS: {fps:.1f}")
            print(f"{'─'*80}\n")
            
            self.last_print_time = current_time
        
        return True
    
    def _on_training_end(self):
        """Print final summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🏁 TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"  Total Episodes:  {self.episode_count}")
        print(f"  Total Steps:     {self.current_timestep:,}")
        print(f"  Total Time:      {total_time/60:.1f} minutes")
        print(f"  Final Stage:     {self.curriculum.get_current_stage()}")
        print(f"  Successes:       {self.success_count}")
        print(f"  Crashes:         {self.crash_count}")
        print(f"  Flips:           {self.flip_count}")
        print(f"{'='*80}\n")

def load_pretrained_policy(model):
    """
    Load imitation learning weights into custom feature extractor.
    
    FIXES:
    1. Added map_location to handle CPU/GPU properly
    2. Added better error messages
    """
    try:
        device = model.policy.device
        print(f"🖥️  PPO is using device: {device}")
        
        if os.path.exists('data/pid_pretrained_policy_best.pth'):
            pretrained = torch.load('data/pid_pretrained_policy_best.pth', map_location=device)
            print("✅ Loading best pretrained model")
        else:
            pretrained = torch.load('data/pid_pretrained_policy.pth', map_location=device)
            print("✅ Loading final pretrained model")
        
        feature_extractor = model.policy.features_extractor
        
        # Load weights
        feature_extractor.net[0].weight.data = pretrained['net.0.weight']
        feature_extractor.net[0].bias.data = pretrained['net.0.bias']
        feature_extractor.net[2].weight.data = pretrained['net.2.weight']
        feature_extractor.net[2].bias.data = pretrained['net.2.bias']
        
        print(f"✅ Loaded pretrained weights into PPO feature extractor on {device}")
        
    except Exception as e:
        print(f"⚠️  Could not load pretrained weights: {e}")
        print("   Starting from scratch...")

def train_full_curriculum():
    """Train PPO through all 3 curriculum stages"""
    
    print("\n" + "="*80)
    print("🎓 FULL CURRICULUM TRAINING")
    print("="*80)
    print("Stage 1: Hover (gentle nudges)")
    print("Stage 2: Disturbance (strong pushes)")
    print("Stage 3: Impact Recovery (violent flips)")
    print("="*80 + "\n")
    
    # Initialize
    curriculum = CurriculumManager()
    env = AirSimDroneEnv(stage=1)
    
    # Define policy kwargs with custom network
    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(),
        net_arch=dict(pi=[256], vf=[256])
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        tensorboard_log="./ppo_logs/",
        policy_kwargs=policy_kwargs
    )
    
    print("\n📐 PPO Architecture:")
    print(f"   Features Extractor: CustomNetwork (13 → 256 → 256)")
    print(f"   Policy Head: 256 → 256 → 4")
    print(f"   Value Head: 256 → 256 → 1")
    print(f"   Total Trainable Params: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Load pretrained policy
    load_pretrained_policy(model)
    
    # Training configuration
    total_timesteps = 500000
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Total Timesteps: {total_timesteps:,}")
    print(f"   Rollout Steps: 2,048")
    print(f"   Batch Size: 64")
    print(f"   Learning Rate: 3e-4")
    print(f"   Expected Episodes: ~{total_timesteps // 250}")
    
    # Create detailed callback
    callback = DetailedProgressCallback(curriculum, env, total_timesteps)
    
    # Train
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Save final model
    final_stage = curriculum.get_current_stage()
    model.save(f"data/ppo_stage{final_stage}_final")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print(f"Final Stage: {final_stage}")
    print(f"Model saved to: data/ppo_stage{final_stage}_final.zip")
    print("="*80)
    
    return model

if __name__ == "__main__":
    train_full_curriculum()