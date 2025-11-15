from stable_baselines3 import PPO
from airsim_env import AirSimDroneEnv
import numpy as np
import time
import os

def test_stage(stage, model_path, num_episodes=3):
    """Test a specific stage"""
    
    print(f"\n{'='*60}")
    print(f"🧪 TESTING STAGE {stage}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    env = AirSimDroneEnv(stage=stage)
    model = PPO.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    success_count = 0
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs = env.reset()
        episode_reward = 0
        max_ang_vel = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            ang_vel = np.linalg.norm(obs[10:13])
            max_ang_vel = max(max_ang_vel, ang_vel)
            
            if step in [100, 200, 300, 400]:
                print(f"  💥 [Step {step}] Injecting disturbance...")
                env._inject_stage_appropriate_disturbance()
            
            if done:
                print(f"  🛑 Episode ended at step {step}")
                break
            
            time.sleep(0.05)
        
        total_rewards.append(episode_reward)
        
        if not done:
            success_count += 1
            print(f"  ✅ Episode SUCCESS")
        else:
            print(f"  ❌ Episode FAILED")
        
        print(f"  📊 Reward: {episode_reward:.2f}")
        print(f"  📊 Max Angular Vel: {max_ang_vel:.2f} rad/s")
    
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / num_episodes) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 STAGE {stage} SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'='*60}\n")

def test_all_stages():
    """Test all curriculum stages"""
    
    print("\n" + "="*60)
    print("🧪 TESTING ALL CURRICULUM STAGES")
    print("="*60)
    
    stages = [
        (1, "data/ppo_stage1_checkpoint.zip"),
        (2, "data/ppo_stage2_checkpoint.zip"),
        (3, "data/ppo_stage3_final.zip")
    ]
    
    for stage, model_path in stages:
        test_stage(stage, model_path, num_episodes=3)
        time.sleep(2)

if __name__ == "__main__":
    test_all_stages()