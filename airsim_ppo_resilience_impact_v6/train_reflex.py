from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
from datetime import datetime
from drone_fast_reflex_env import DroneReflexRecoveryEnv

class ReflexMetricsCallback(BaseCallback):
    """Track recovery speed metrics"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.recovery_times = []
        self.sub_500ms_recoveries = 0
        self.sub_1s_recoveries = 0
        self.total_recoveries = 0
        
    def _on_step(self):
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode_stats' in info:
                stats = info['episode_stats']
                
                if stats['successful_recoveries'] > 0 and stats['recovery_time_ms'] > 0:
                    recovery_ms = stats['recovery_time_ms']
                    self.recovery_times.append(recovery_ms)
                    self.total_recoveries += 1
                    
                    if recovery_ms < 500:
                        self.sub_500ms_recoveries += 1
                    if recovery_ms < 1000:
                        self.sub_1s_recoveries += 1
        
        return True
    
    def _on_training_end(self):
        if self.recovery_times:
            print(f"\n⚡ REFLEX PERFORMANCE:")
            print(f"   Total recoveries: {self.total_recoveries}")
            print(f"   Average time: {np.mean(self.recovery_times):.0f}ms")
            print(f"   Fastest: {np.min(self.recovery_times):.0f}ms")
            print(f"   Under 500ms: {self.sub_500ms_recoveries} ({self.sub_500ms_recoveries/self.total_recoveries*100:.1f}%)")
            print(f"   Under 1000ms: {self.sub_1s_recoveries} ({self.sub_1s_recoveries/self.total_recoveries*100:.1f}%)")

class CurriculumCallback(BaseCallback):
    """Auto-advance stages"""
    def __init__(self, env, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.current_stage = 1
        self.episode_rewards = []
        self.recovery_successes = []
        
    def _on_step(self):
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                
            if 'episode_stats' in info:
                stats = info['episode_stats']
                success = stats['successful_recoveries'] > 0 and stats['crashes'] == 0
                self.recovery_successes.append(float(success))
        
        # Check advancement every 100 episodes
        if len(self.episode_rewards) >= 100 and len(self.episode_rewards) % 100 == 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_success = self.recovery_successes[-100:]
            
            mean_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_success)
            
            # Advance if performing well
            if self.current_stage < 5:
                should_advance = False
                
                if self.current_stage == 1 and mean_reward > 40 and success_rate > 0.7:
                    should_advance = True
                elif self.current_stage == 2 and mean_reward > 20 and success_rate > 0.5:
                    should_advance = True
                elif self.current_stage == 3 and mean_reward > 10 and success_rate > 0.4:
                    should_advance = True
                elif self.current_stage == 4 and mean_reward > 5 and success_rate > 0.3:
                    should_advance = True
                
                if should_advance:
                    self.current_stage += 1
                    print(f"\n🎓 ADVANCING TO STAGE {self.current_stage}!")
                    print(f"   Reward: {mean_reward:.1f} | Success: {success_rate*100:.1f}%\n")
                    
                    if hasattr(self.env, 'env_method'):
                        self.env.env_method('set_training_stage', self.current_stage)
                    else:
                        self.env.set_training_stage(self.current_stage)
        
        return True

def make_env():
    def _init():
        env = DroneReflexRecoveryEnv()
        return Monitor(env)
    return _init

def train_reflex_recovery(total_timesteps=2000000):
    """Train ultra-fast recovery"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./models/reflex_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    print("\n" + "="*70)
    print("⚡ ULTRA-FAST REFLEX RECOVERY TRAINING ⚡")
    print("="*70)
    print("\nGoal: Sub-second recovery from violent flips")
    print("Control: 50Hz (20ms steps)")
    print("Target: <500ms recovery time")
    print("="*70)
    
    # Create environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Callbacks
    reflex_callback = ReflexMetricsCallback(verbose=1)
    curriculum_callback = CurriculumCallback(env, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="ppo_reflex"
    )
    
    # Create PPO - optimized for high-frequency control
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Higher LR for faster learning
        n_steps=1024,  # Smaller due to faster steps
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
            net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])]  # Larger network
        )
    )
    
    print(f"\n🚀 Training for {total_timesteps:,} steps...")
    print(f"💾 Saving to: {save_dir}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[reflex_callback, curriculum_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted")
    
    # Save
    model.save(os.path.join(save_dir, "ppo_reflex_final"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\n✅ Training complete! Saved to: {save_dir}")
    
    return model, env, save_dir

def test_reflex(model_path, vec_norm_path, num_episodes=20):
    """Test reflex recovery speed"""
    print("\n⚡ Testing Reflex Recovery Speed...\n")
    
    model = PPO.load(model_path)
    
    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    
    recovery_times_ms = []
    crashes = 0
    successful = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*70}")
        
        while not done and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1
        
        if len(info) > 0 and 'episode_stats' in info[0]:
            stats = info[0]['episode_stats']
            
            if stats['successful_recoveries'] > 0:
                recovery_ms = stats['recovery_time_ms']
                recovery_times_ms.append(recovery_ms)
                successful += 1
                
                if recovery_ms < 300:
                    print(f"🚀 LIGHTNING: {recovery_ms:.0f}ms")
                elif recovery_ms < 500:
                    print(f"⚡ VERY FAST: {recovery_ms:.0f}ms")
                elif recovery_ms < 1000:
                    print(f"✅ FAST: {recovery_ms:.0f}ms")
                else:
                    print(f"✓ Recovered: {recovery_ms:.0f}ms")
            
            if stats['crashes'] > 0:
                crashes += 1
                print(f"💥 CRASHED")
    
    # Results
    print(f"\n{'='*70}")
    print(f"⚡ REFLEX TEST RESULTS")
    print(f"{'='*70}")
    
    if recovery_times_ms:
        sub_300 = sum(1 for t in recovery_times_ms if t < 300)
        sub_500 = sum(1 for t in recovery_times_ms if t < 500)
        sub_1000 = sum(1 for t in recovery_times_ms if t < 1000)
        
        print(f"✅ Successful: {successful}/{num_episodes} ({successful/num_episodes*100:.1f}%)")
        print(f"💥 Crashed: {crashes}/{num_episodes}")
        print(f"\n⏱️  Recovery Times:")
        print(f"   Average: {np.mean(recovery_times_ms):.0f}ms")
        print(f"   Median: {np.median(recovery_times_ms):.0f}ms")
        print(f"   Fastest: {np.min(recovery_times_ms):.0f}ms")
        print(f"   Slowest: {np.max(recovery_times_ms):.0f}ms")
        print(f"\n🎯 Speed Distribution:")
        print(f"   <300ms:  {sub_300} ({sub_300/len(recovery_times_ms)*100:.1f}%)")
        print(f"   <500ms:  {sub_500} ({sub_500/len(recovery_times_ms)*100:.1f}%)")
        print(f"   <1000ms: {sub_1000} ({sub_1000/len(recovery_times_ms)*100:.1f}%)")
    
    env.close()

if __name__ == "__main__":
    print("\n⚡ ULTRA-FAST REFLEX TRAINING ⚡")
    print("Goal: Sub-second recovery from violent flips\n")
    
    choice = input("1) Train  2) Test: ").strip()
    
    if choice == "1":
        timesteps = input("Timesteps (default 2000000): ").strip()
        timesteps = int(timesteps) if timesteps else 2000000
        train_reflex_recovery(total_timesteps=timesteps)
    else:
        model_path = input("Model path: ").strip()
        vec_norm_path = input("VecNormalize path: ").strip()
        episodes = input("Episodes (default 20): ").strip()
        episodes = int(episodes) if episodes else 20
        test_reflex(model_path, vec_norm_path, episodes)