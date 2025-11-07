"""
TEST HOVER POLICY
=================
Tests the trained neural network policy (not PID expert!) to see if it learned to hover.

This uses the NEURAL NETWORK that was trained via behavioral cloning.
Success criteria: 80%+ episodes should complete without crashing.
"""

import torch
import torch.nn as nn
import numpy as np
import airsim
import time
import pickle
from pathlib import Path


class HoverPolicy(nn.Module):
    """Same architecture as training"""
    
    def __init__(self, state_dim=10, action_dim=3):
        super(HoverPolicy, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def predict(self, state):
        """Predict action for a single state"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            return self.network(state).squeeze(0).numpy()


def get_observation(client):
    """Get current observation from drone"""
    drone_state = client.getMultirotorState()
    
    pos = drone_state.kinematics_estimated.position
    vel = drone_state.kinematics_estimated.linear_velocity
    ori = drone_state.kinematics_estimated.orientation
    
    obs = np.array([
        pos.x_val, pos.y_val, pos.z_val,
        vel.x_val, vel.y_val, vel.z_val,
        ori.w_val, ori.x_val, ori.y_val, ori.z_val
    ], dtype=np.float32)
    
    return obs, pos, vel


def test_policy(model_path, num_episodes=10, max_steps=500, target_alt=10.0):
    """Test the learned policy"""
    
    print("\n" + "="*70)
    print("🧪 TESTING LEARNED HOVER POLICY")
    print("="*70)
    print("This uses the NEURAL NETWORK, not the PID expert!")
    print()
    
    # Load model
    print(f"[1/3] Loading model: {model_path}")
    model = HoverPolicy()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("   ✅ Model loaded successfully")
    print()
    
    # Connect to AirSim
    print("[2/3] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("   ✅ Connected")
    print()
    
    # Test episodes
    print(f"[3/3] Running {num_episodes} test episodes...")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Target altitude: {target_alt}m")
    print()
    print("="*70)
    
    results = []
    
    for episode in range(1, num_episodes + 1):
        # Reset
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        # Takeoff
        client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Move to starting position
        client.moveToPositionAsync(0, 0, -target_alt, 5).join()
        time.sleep(1.0)
        
        # Episode variables
        episode_reward = 0
        distances = []
        crashed = False
        reason = "completed"
        
        # Run episode
        for step in range(max_steps):
            # Get observation
            obs, pos, vel = get_observation(client)
            
            # Neural network predicts action
            action = model.predict(obs)
            
            # Execute action
            client.moveByVelocityAsync(
                float(action[0]),
                float(action[1]),
                float(action[2]),
                duration=0.05,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
            
            # Calculate metrics
            alt = -pos.z_val
            dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
            dist_from_target_alt = abs(alt - target_alt)
            
            distances.append(dist_from_center)
            
            # Check termination
            if dist_from_center > 20:
                reason = "out_of_bounds"
                crashed = True
                break
            
            if alt < 2 or alt > 30:
                reason = "altitude_violation"
                crashed = True
                break
            
            # Collision check
            collision = client.simGetCollisionInfo()
            if collision.has_collided:
                reason = "collision"
                crashed = True
                break
        
        # Episode results
        success = not crashed
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        results.append({
            'episode': episode,
            'success': success,
            'steps': step + 1,
            'avg_distance': avg_distance,
            'max_distance': max_distance,
            'reason': reason
        })
        
        # Print episode result
        status = "✅" if success else "❌"
        print(f"Episode {episode:2d}/{num_episodes} | Steps: {step+1:3d} | "
              f"Success: {status} | Avg Dist: {avg_distance:.2f}m | Reason: {reason}")
    
    print("="*70)
    
    # Overall statistics
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / num_episodes * 100
    
    avg_steps = np.mean([r['steps'] for r in results])
    avg_distance = np.mean([r['avg_distance'] for r in results if r['success']])
    
    print(f"Success Rate: {success_rate:.0f}% ({successes}/{num_episodes} episodes)")
    print(f"Average Episode Length: {avg_steps:.1f} steps")
    
    if successes > 0:
        print(f"Average Distance from Target: {avg_distance:.2f}m (successful episodes)")
    
    print()
    
    # Failure analysis
    if successes < num_episodes:
        print("Failure Reasons:")
        reasons = {}
        for r in results:
            if not r['success']:
                reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        for reason, count in reasons.items():
            print(f"   - {reason}: {count}")
        print()
    
    # Verdict
    print("="*70)
    if success_rate >= 80:
        print("✅ EXCELLENT! Policy successfully learned to hover!")
        print("   Ready for Stage 2 (disturbance recovery)")
    elif success_rate >= 60:
        print("⚠️  GOOD! Policy learned to hover but could be better.")
        print("   Consider collecting more data (2000 episodes)")
    elif success_rate >= 40:
        print("⚠️  MODERATE! Policy partially learned.")
        print("   Recommend: Collect 2000 episodes and retrain")
    else:
        print("❌ POOR! Policy did not learn well.")
        print("   Action needed:")
        print("   1. Check PID expert is tuned well")
        print("   2. Collect more data (2000 episodes)")
        print("   3. Train for more epochs (200)")
    print("="*70 + "\n")
    
    # Cleanup
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./models/hover_policy_best.pth',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--altitude', type=float, default=10.0,
                        help='Target hover altitude')
    
    args = parser.parse_args()
    
    results = test_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.steps,
        target_alt=args.altitude
    )