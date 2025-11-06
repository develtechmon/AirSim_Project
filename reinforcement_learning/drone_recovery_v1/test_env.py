"""
Quick test script to verify environment works correctly.
Run this BEFORE training to catch issues early.
"""

from airsim_env import AirSimDroneEnv
import time
import numpy as np

def test_environment():
    print("\n" + "="*80)
    print("🧪 ENVIRONMENT TEST")
    print("="*80 + "\n")
    
    env = AirSimDroneEnv(stage=1)
    
    print("Testing 3 episodes with random actions...\n")
    
    for episode in range(3):
        print(f"\n{'─'*80}")
        print(f"Episode {episode + 1}:")
        print(f"{'─'*80}")
        
        obs, info = env.reset()
        print(f"  ✅ Reset complete")
        print(f"  Initial position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
        print(f"  Initial orientation: qw={obs[3]:.3f}")
        
        episode_reward = 0
        step_count = 0
        
        for step in range(100):
            # Random action from action space
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Print first 3 steps
            if step < 3:
                print(f"  Step {step}: pos=({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}), reward={reward:.2f}")
            
            if terminated or truncated:
                print(f"\n  Episode ended:")
                print(f"    Steps: {step_count}")
                print(f"    Total reward: {episode_reward:.2f}")
                print(f"    Final position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")
                
                if terminated:
                    if obs[2] > -0.5:
                        print(f"    Reason: CRASHED (z={obs[2]:.2f})")
                    elif obs[3] < 0.3:
                        print(f"    Reason: FLIPPED (qw={obs[3]:.3f})")
                    elif np.linalg.norm(obs[0:2]) > 20:
                        print(f"    Reason: OUT OF BOUNDS (dist={np.linalg.norm(obs[0:2]):.2f})")
                elif truncated:
                    print(f"    Reason: MAX STEPS (500)")
                
                break
        
        time.sleep(1)
    
    print(f"\n{'='*80}")
    print("✅ ENVIRONMENT TEST COMPLETE")
    print("="*80 + "\n")
    
    print("If you saw:")
    print("  - Episodes with >0 steps: ✅ Environment is working")
    print("  - Crashes after some steps: ✅ Normal for random actions")
    print("  - Episodes with 0 steps: ❌ Bug in environment")
    print("\nYou can now run: python train_ppo.py")

if __name__ == "__main__":
    test_environment()