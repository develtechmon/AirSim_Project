"""
COLLECT EXPERT DEMONSTRATIONS
==============================
Runs PID expert controller and saves state-action pairs
for behavioral cloning (imitation learning).

Goal: Collect 2000 episodes of perfect hovering
Time: ~30-60 minutes

Dataset size: ~400,000 state-action pairs

python collect_demonstrations.py --episodes 100 --steps 200 (For test 1st)
python collect_demonstrations.py --episodes 2000 (60 min)

python collect_demonstration_v2.py --episodes 200 --steps 200 

py .\stage_1\collect_demonstration.py --episodes 600 --steps 200 (i'm using this command to generate 120,000 samples)

Time with your speed:

200 episodes ÷ 3 eps/min = ~65 minutes ✅
Samples: 40,000 (still good for learning!)

*************** Final Result *****************
======================================================================
📊 COLLECTING EXPERT DEMONSTRATIONS
======================================================================
Target episodes: 200
Steps per episode: 200
Total data points: ~40,000
Save directory: ./demonstrations
======================================================================

Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

✓ PID Expert Controller Initialized (13 observations)
  Target: Hover at 30.0m
  Control frequency: 20 Hz
  Observation space: 13 (includes angular velocity)
Starting collection...

Episode   10/200 | Avg Reward:  5834.8 | Speed: 2.5 eps/min | ETA: 75 min
Episode   20/200 | Avg Reward:  5831.7 | Speed: 2.5 eps/min | ETA: 71 min
Episode   30/200 | Avg Reward:  5850.8 | Speed: 2.5 eps/min | ETA: 67 min
Episode   40/200 | Avg Reward:  5838.5 | Speed: 2.5 eps/min | ETA: 63 min
Episode   50/200 | Avg Reward:  5836.4 | Speed: 2.5 eps/min | ETA: 59 min
Episode   60/200 | Avg Reward:  5828.3 | Speed: 2.5 eps/min | ETA: 55 min
Episode   70/200 | Avg Reward:  5851.4 | Speed: 2.5 eps/min | ETA: 51 min
Episode   80/200 | Avg Reward:  5869.0 | Speed: 2.5 eps/min | ETA: 47 min
Episode   90/200 | Avg Reward:  5828.0 | Speed: 2.5 eps/min | ETA: 43 min
Episode  100/200 | Avg Reward:  5832.9 | Speed: 2.5 eps/min | ETA: 39 min
Episode  110/200 | Avg Reward:  5843.3 | Speed: 2.5 eps/min | ETA: 36 min
Episode  120/200 | Avg Reward:  5833.0 | Speed: 2.5 eps/min | ETA: 32 min
Episode  130/200 | Avg Reward:  5855.7 | Speed: 2.5 eps/min | ETA: 28 min
Episode  140/200 | Avg Reward:  5828.0 | Speed: 2.5 eps/min | ETA: 24 min
Episode  150/200 | Avg Reward:  5823.7 | Speed: 2.5 eps/min | ETA: 20 min
Episode  160/200 | Avg Reward:  5836.7 | Speed: 2.5 eps/min | ETA: 16 min
Episode  170/200 | Avg Reward:  5824.2 | Speed: 2.5 eps/min | ETA: 12 min
Episode  180/200 | Avg Reward:  5838.2 | Speed: 2.5 eps/min | ETA: 8 min
Episode  190/200 | Avg Reward:  5856.7 | Speed: 2.5 eps/min | ETA: 4 min
Episode  200/200 | Avg Reward:  5846.6 | Speed: 2.5 eps/min | ETA: 0 min

======================================================================
💾 SAVING FINAL DATASET
======================================================================

📊 Dataset Statistics:
   Total samples: 40,000
   State dimension: 13
   Action dimension: 3
   Mean episode reward: 5839.4
   Std episode reward: 33.5
   Collection time: 79.1 minutes

💾 Saved to: ./demonstrations/expert_demonstrations.pkl
   File size: 4.6 MB
======================================================================


✅ Collection complete!
Next step: Run train_imitation_v2.py to train the policy

"""

import airsim
import numpy as np
import time
import pickle
from pathlib import Path
from pid_expert import PIDExpertHover


def collect_demonstrations(num_episodes=2000, max_steps=200, save_dir="./demonstrations"):
    """
    Collect expert demonstrations
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Steps per episode
        save_dir: Where to save dataset
    """
    print("\n" + "="*70)
    print("📊 COLLECTING EXPERT DEMONSTRATIONS")
    print("="*70)
    print(f"Target episodes: {num_episodes}")
    print(f"Steps per episode: {max_steps}")
    print(f"Total data points: ~{num_episodes * max_steps:,}")
    print(f"Save directory: {save_dir}")
    print("="*70 + "\n")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    # Create PID expert
    expert = PIDExpertHover(target_altitude=30.0)
    
    # Storage
    all_states = []
    all_actions = []
    episode_rewards = []
    
    print("Starting collection...\n")
    start_time = time.time()
    
    # ONE-TIME init before the loop
    client.reset()
    time.sleep(1.0)
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)
    client.takeoffAsync().join()
    time.sleep(1.0)
    client.moveToPositionAsync(0, 0, -30, 5, timeout_sec=20).join()
    time.sleep(1.0)
    print("Drone at 30m. Starting episodes...\n")

    for episode in range(num_episodes):
        # # Reset
        # client.reset()
        # client.enableApiControl(True)
        # client.armDisarm(True)
        
        # # Takeoff
        # client.takeoffAsync().join()
        # time.sleep(0.5)
        
        # Move to starting position with small random offset
        # start_x = np.random.uniform(-2, 2)
        # start_y = np.random.uniform(-2, 2)
        # client.moveToPositionAsync(start_x, start_y, -30, 5).join()
        # time.sleep(1.0)
        
        # Reposition between episodes using pose setter (no reset needed)
        start_x = np.random.uniform(-2, 2)
        start_y = np.random.uniform(-2, 2)
        if episode > 0:
            pose = airsim.Pose(
                airsim.Vector3r(start_x, start_y, -30),
                airsim.to_quaternion(0, 0, 0)
            )
            client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(0.3)
            client.enableApiControl(True)
            client.armDisarm(True)
            time.sleep(0.3)
        
        # Reset expert
        expert.reset()
        
        # Collect trajectory
        episode_states = []
        episode_actions = []
        episode_reward = 0
        
        for step in range(max_steps):
            # Get state
            drone_state = client.getMultirotorState()
            pos = drone_state.kinematics_estimated.position
            vel = drone_state.kinematics_estimated.linear_velocity
            ori = drone_state.kinematics_estimated.orientation
            ang_vel = drone_state.kinematics_estimated.angular_velocity  # ← ONLY NEW LINE!
            
            # Create state dict
            state = {
                'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
                'velocity': np.array([vel.x_val, vel.y_val, vel.z_val]),
                'orientation': np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val]),
                'angular_velocity': np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])  # ← ONLY NEW LINE!
            }
            
            # Get expert action
            action = expert.get_action(state)
            
            # Execute
            client.moveByVelocityAsync(
                float(action[0]), float(action[1]), float(action[2]),
                duration=expert.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
            
            # Create observation for neural network (flatten state)
            obs = np.concatenate([
                state['position'],
                state['velocity'],
                state['orientation'],
                state['angular_velocity']  # ← ONLY NEW LINE!
            ])
            
            # Calculate reward (for statistics)
            alt = -pos.z_val
            dist_from_center = np.linalg.norm([pos.x_val, pos.y_val])
            dist_from_target_alt = abs(alt - 30.0)
            reward = 30.0 - dist_from_target_alt - dist_from_center * 0.5
            episode_reward += reward
            
            # Store
            episode_states.append(obs)
            episode_actions.append(action)
        
        # Add to dataset
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        episode_rewards.append(episode_reward)
        
        # Progress
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:])
            eps_per_min = (episode + 1) / (elapsed / 60)
            eta_min = (num_episodes - episode - 1) / eps_per_min
            
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Speed: {eps_per_min:.1f} eps/min | "
                  f"ETA: {eta_min:.0f} min")
        
        # Save checkpoint every 500 episodes
        if (episode + 1) % 500 == 0:
            checkpoint_path = f"{save_dir}/checkpoint_{episode+1}.pkl"
            dataset = {
                'states': np.array(all_states),
                'actions': np.array(all_actions),
                'rewards': episode_rewards
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Final save
    print("\n" + "="*70)
    print("💾 SAVING FINAL DATASET")
    print("="*70)
    
    dataset = {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'rewards': episode_rewards,
        'metadata': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'total_samples': len(all_states),
            'state_dim': all_states[0].shape[0],
            'action_dim': all_actions[0].shape[0],
            'collection_time': time.time() - start_time
        }
    }
    
    final_path = f"{save_dir}/expert_demonstrations.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(all_states):,}")
    print(f"   State dimension: {all_states[0].shape[0]}")
    print(f"   Action dimension: {all_actions[0].shape[0]}")
    print(f"   Mean episode reward: {np.mean(episode_rewards):.1f}")
    print(f"   Std episode reward: {np.std(episode_rewards):.1f}")
    print(f"   Collection time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"\n💾 Saved to: {final_path}")
    print(f"   File size: {Path(final_path).stat().st_size / 1e6:.1f} MB")
    print("="*70 + "\n")
    
    # Cleanup
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
    parser.add_argument('--save-dir', type=str, default='./demonstrations', help='Save directory')
    args = parser.parse_args()
    
    dataset = collect_demonstrations(
        num_episodes=args.episodes,
        max_steps=args.steps,
        save_dir=args.save_dir
    )
    
    print("\n✅ Collection complete!")
    print("Next step: Run train_imitation_v2.py to train the policy")