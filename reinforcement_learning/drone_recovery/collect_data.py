import airsim
import numpy as np
import time
import os

def collect_hover_data(num_episodes=50, steps_per_episode=200):
    """
    Collect hover data with periodic disturbances.
    COMPLETE FIXED VERSION with proper normalization.
    """
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    dataset = {'obs': [], 'actions': []}
    
    print("🚁 Starting data collection...")
    print(f"📊 Collecting {num_episodes} episodes with {steps_per_episode} steps each")
    
    # TEST: Check if rotor states are available
    print("\n🔍 Testing rotor state availability...")
    test_rotor = client.getRotorStates()
    
    # Determine rotor format
    ROTOR_FORMAT = None
    if hasattr(test_rotor, 'rotors'):
        rotor_list = test_rotor.rotors
        if len(rotor_list) == 0:
            print("❌ ERROR: No rotors found!")
            return None
        
        first_rotor = rotor_list[0]
        if isinstance(first_rotor, dict):
            print(f"✅ Rotor states available: {len(rotor_list)} rotors (dict format)")
            print(f"   Sample speeds: {[r['speed'] for r in rotor_list[:4]]}")
            ROTOR_FORMAT = 'dict'
        else:
            print(f"✅ Rotor states available: {len(rotor_list)} rotors (object format)")
            print(f"   Sample speeds: {[r.speed for r in rotor_list[:4]]}")
            ROTOR_FORMAT = 'object'
    else:
        print("❌ ERROR: Unexpected rotor state structure!")
        print(f"   Got: {type(test_rotor)}")
        return None
    
    def get_motor_speeds(rotor_states):
        """Extract motor speeds regardless of format"""
        if not hasattr(rotor_states, 'rotors') or len(rotor_states.rotors) == 0:
            return None
        
        if ROTOR_FORMAT == 'dict':
            return np.array([r['speed'] for r in rotor_states.rotors])
        else:
            return np.array([r.speed for r in rotor_states.rotors])
    
    for episode in range(num_episodes):
        print(f"\n🔄 Episode {episode + 1}/{num_episodes}")
        
        # Reset to starting position
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        # Takeoff and wait for stabilization
        print("  ⬆️  Taking off...")
        client.takeoffAsync().join()
        time.sleep(3)
        
        # Move to hover position and WAIT
        target_pos = airsim.Vector3r(0, 0, -10)
        print(f"  🎯 Moving to hover position...")
        client.moveToPositionAsync(
            target_pos.x_val, 
            target_pos.y_val, 
            target_pos.z_val, 
            velocity=1
        ).join()
        
        time.sleep(2)
        print("  ✅ Hovering stable, starting collection...")
        
        for step in range(steps_per_episode):
            # Get current state
            state = client.getMultirotorState()
            
            # Create observation vector (13 dimensions)
            obs = np.array([
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val,
                state.kinematics_estimated.orientation.w_val,
                state.kinematics_estimated.orientation.x_val,
                state.kinematics_estimated.orientation.y_val,
                state.kinematics_estimated.orientation.z_val,
                state.kinematics_estimated.linear_velocity.x_val,
                state.kinematics_estimated.linear_velocity.y_val,
                state.kinematics_estimated.linear_velocity.z_val,
                state.kinematics_estimated.angular_velocity.x_val,
                state.kinematics_estimated.angular_velocity.y_val,
                state.kinematics_estimated.angular_velocity.z_val,
            ])
            
            # Get rotor states
            rotor_states = client.getRotorStates()
            motor_speeds = get_motor_speeds(rotor_states)
            
            if motor_speeds is None:
                print(f"  ⚠️  Warning: No rotor data at step {step}, skipping...")
                continue
            
            # Diagnostic on first sample
            if episode == 0 and step == 0:
                print(f"\n📊 DIAGNOSTIC - Raw rotor speeds (RPM): {motor_speeds}")
            
            # Convert to thrust (proportional to speed^2)
            # Keep as raw values - normalize after collection
            motor_thrusts = motor_speeds ** 2
            
            dataset['obs'].append(obs)
            dataset['actions'].append(motor_thrusts)
            
            # Inject disturbance every 50 steps
            if step % 50 == 0 and step > 0:
                disturbance_strength = np.random.uniform(1.0, 3.0)
                
                pose = client.simGetVehiclePose()
                pose.position.x_val += np.random.uniform(-disturbance_strength, disturbance_strength)
                pose.position.y_val += np.random.uniform(-disturbance_strength, disturbance_strength)
                
                # Add rotation to simulate tumbling
                try:
                    from scipy.spatial.transform import Rotation as R
                    
                    # Current orientation
                    current_rot = R.from_quat([
                        pose.orientation.x_val,
                        pose.orientation.y_val,
                        pose.orientation.z_val,
                        pose.orientation.w_val
                    ])
                    
                    # Add random rotation (small angles for Stage 1)
                    delta_rot = R.from_euler('xyz', [
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.2, 0.2)
                    ])
                    
                    # Combine rotations
                    new_rot = current_rot * delta_rot
                    new_quat = new_rot.as_quat()
                    
                    pose.orientation = airsim.Quaternionr(
                        new_quat[0], new_quat[1], new_quat[2], new_quat[3]
                    )
                except ImportError:
                    print("  ⚠️  scipy not installed, skipping rotation disturbance")
                
                client.simSetVehiclePose(pose, True)
                print(f"  💥 Injected disturbance at step {step}")
                
                # Re-engage position hold to let PID recover
                client.moveToPositionAsync(
                    target_pos.x_val,
                    target_pos.y_val,
                    target_pos.z_val,
                    velocity=1
                )
            
            time.sleep(0.1)
        
        print(f"  ✅ Collected {len(dataset['obs'])} samples so far")
    
    # Validate dataset
    if len(dataset['obs']) == 0:
        print("\n❌ ERROR: No data collected!")
        return None
    
    # Convert to arrays
    print(f"\n📊 Raw Dataset Statistics (before normalization):")
    obs_array = np.array(dataset['obs'])
    actions_array = np.array(dataset['actions'])
    
    print(f"   Observations shape: {obs_array.shape}")
    print(f"   Actions shape: {actions_array.shape}")
    print(f"   Raw action mean: {actions_array.mean(axis=0)}")
    print(f"   Raw action std: {actions_array.std(axis=0)}")
    print(f"   Raw action range: [{actions_array.min():.2e}, {actions_array.max():.2e}]")
    
    # Normalize actions to zero-mean, unit-variance
    action_mean = actions_array.mean(axis=0)
    action_std = actions_array.std(axis=0)
    
    # Prevent division by zero
    action_std = np.where(action_std < 1e-6, 1.0, action_std)
    
    actions_normalized = (actions_array - action_mean) / action_std
    
    # Save normalization params (needed for deployment)
    normalization_params = {
        'action_mean': action_mean,
        'action_std': action_std
    }
    
    print(f"\n📊 Normalized Dataset Statistics:")
    print(f"   Action mean: {actions_normalized.mean(axis=0)}")
    print(f"   Action std: {actions_normalized.std(axis=0)}")
    print(f"   Action range: [{actions_normalized.min():.2f}, {actions_normalized.max():.2f}]")
    
    # Update dataset
    dataset['actions'] = actions_normalized.tolist()
    
    # Check for NaN or Inf
    if np.isnan(obs_array).any() or np.isinf(obs_array).any():
        print("\n⚠️  WARNING: Dataset contains NaN or Inf in observations!")
    
    if np.isnan(actions_normalized).any() or np.isinf(actions_normalized).any():
        print("\n⚠️  WARNING: Dataset contains NaN or Inf in actions!")
    
    # Save dataset and params
    os.makedirs('data', exist_ok=True)
    np.save('data/hover_dataset.npy', dataset)
    np.save('data/normalization_params.npy', normalization_params)
    
    print(f"\n✅ Data collection complete!")
    print(f"📦 Saved {len(dataset['obs'])} samples to data/hover_dataset.npy")
    print(f"📦 Saved normalization params to data/normalization_params.npy")
    print(f"📊 Observation shape: {obs_array.shape}")
    print(f"🎮 Action shape: {actions_normalized.shape}")
    
    return dataset

if __name__ == "__main__":
    # Make sure AirSim is running!
    dataset = collect_hover_data(num_episodes=10, steps_per_episode=200)
    
    if dataset is not None:
        print("\n✅ SUCCESS! Ready for imitation learning.")
    else:
        print("\n❌ FAILED! Check errors above.")