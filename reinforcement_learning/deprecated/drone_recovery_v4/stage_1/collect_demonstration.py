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

Result
======================================================================
    SAVING FINAL DATASET
======================================================================

    Dataset Statistics:
    Total samples: 20,000
    State dimension: 10
    Action dimension: 3
    Mean episode reward: 1811.8
    Std episode reward: 33.8
    Collection time: 32.8 minutes

    Saved to: ./demonstrations/expert_demonstrations.pkl
    File size: 1.8 MB
=======================
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
    expert = PIDExpertHover(target_altitude=10.0)
    
    # Storage
    all_states = []
    all_actions = []
    episode_rewards = []
    
    print("Starting collection...\n")
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        # Takeoff
        client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Move to starting position with small random offset
        start_x = np.random.uniform(-2, 2)
        start_y = np.random.uniform(-2, 2)
        client.moveToPositionAsync(start_x, start_y, -10, 5).join()
        time.sleep(1.0)
        
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
            
            # Create state dict
            state = {
                'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
                'velocity': np.array([vel.x_val, vel.y_val, vel.z_val]),
                'orientation': np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
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
                state['orientation']
            ])
            
            # Calculate reward (for statistics)
            alt = -pos.z_val
            dist_from_center = np.linalg.norm([pos.x_val, pos.y_val])
            dist_from_target_alt = abs(alt - 10.0)
            reward = 10.0 - dist_from_target_alt - dist_from_center * 0.5
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
    print("Next step: Run train_imitation.py to train the policy")