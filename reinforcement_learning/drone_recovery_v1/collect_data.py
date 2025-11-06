# collect_data.py
import airsim
import numpy as np
import time
import os

def collect_hover_data(num_episodes=50, steps_per_episode=200):
    """
    Collect hover data with periodic disturbances.
    Records actions as *physical PWM [0,1]* (NO z-scoring).
    """

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    dataset = {'obs': [], 'actions': []}

    print("🚁 Starting data collection...")
    print(f"📊 Collecting {num_episodes} episodes × {steps_per_episode} steps")

    # Probe rotor interface
    print("\n🔍 Testing rotor state availability...")
    test_rotor = client.getRotorStates()

    ROTOR_FORMAT = None
    if hasattr(test_rotor, 'rotors'):
        rotor_list = test_rotor.rotors
        if len(rotor_list) == 0:
            print("❌ ERROR: No rotors found!")
            return None
        first = rotor_list[0]
        if isinstance(first, dict):
            ROTOR_FORMAT = 'dict'
            sample = [r['speed'] for r in rotor_list[:4]]
        else:
            ROTOR_FORMAT = 'object'
            sample = [r.speed for r in rotor_list[:4]]
        print(f"✅ Rotor states OK: {len(rotor_list)} rotors ({ROTOR_FORMAT}). Sample speeds: {sample}")
    else:
        print("❌ ERROR: Unexpected rotor state structure:", type(test_rotor))
        return None

    def get_motor_speeds(rotor_states):
        if not hasattr(rotor_states, 'rotors') or len(rotor_states.rotors) == 0:
            return None
        if ROTOR_FORMAT == 'dict':
            return np.array([r['speed'] for r in rotor_states.rotors], dtype=np.float32)
        else:
            return np.array([r.speed for r in rotor_states.rotors], dtype=np.float32)

    # Try to determine max rotor RPM
    max_rpm_estimate = None

    for episode in range(num_episodes):
        print(f"\n🔄 Episode {episode + 1}/{num_episodes}")

        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)

        print("  ⬆️  Takeoff...")
        client.takeoffAsync().join()
        time.sleep(2.0)

        target_pos = airsim.Vector3r(0, 0, -10)
        print("  🎯 Move to hover...")
        client.moveToPositionAsync(target_pos.x_val, target_pos.y_val, target_pos.z_val, velocity=1).join()
        time.sleep(2.0)
        print("  ✅ Stable. Start logging...")

        for step in range(steps_per_episode):
            state = client.getMultirotorState()
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
            ], dtype=np.float32)

            # Read rotor speeds
            rotor_states = client.getRotorStates()
            motor_speeds = get_motor_speeds(rotor_states)
            if motor_speeds is None:
                continue

            # Calibrate max RPM once (first episode, first steps)
            if max_rpm_estimate is None:
                print("\n🔍 Calibrating MaxRotorSpeed...")
                try:
                    vehicle_info = client.getVehicleParams()
                    max_rpm_estimate = float(vehicle_info.rotor_configuration['MaxRotorSpeed'])
                    print(f"⚙️  Using MaxRotorSpeed from config: {max_rpm_estimate:.1f} RPM")
                except Exception:
                    # Fallback: percentile of observed speeds
                    all_speeds = []
                    for _ in range(10):
                        s = get_motor_speeds(client.getRotorStates())
                        if s is not None:
                            all_speeds.extend(s.tolist())
                        time.sleep(0.05)
                    max_rpm_estimate = float(np.percentile(all_speeds, 99)) if len(all_speeds) > 0 else 8000.0
                    print(f"⚙️  Estimated MaxRotorSpeed ≈ {max_rpm_estimate:.1f} RPM (observed)")

            # Convert RPM → PWM [0,1]
            motor_pwms = np.clip(motor_speeds / max_rpm_estimate, 0.0, 1.0).astype(np.float32)

            dataset['obs'].append(obs)
            dataset['actions'].append(motor_pwms)

            # Disturbance every 50 steps
            if step > 0 and step % 50 == 0:
                disturbance_strength = float(np.random.uniform(1.0, 3.0))
                pose = client.simGetVehiclePose()
                pose.position.x_val += np.random.uniform(-disturbance_strength, disturbance_strength)
                pose.position.y_val += np.random.uniform(-disturbance_strength, disturbance_strength)
                try:
                    from scipy.spatial.transform import Rotation as R
                    current_rot = R.from_quat([
                        pose.orientation.x_val,
                        pose.orientation.y_val,
                        pose.orientation.z_val,
                        pose.orientation.w_val
                    ])
                    delta_rot = R.from_euler('xyz', [
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.2, 0.2)
                    ])
                    new_quat = (current_rot * delta_rot).as_quat()
                    pose.orientation = airsim.Quaternionr(new_quat[0], new_quat[1], new_quat[2], new_quat[3])
                except Exception:
                    pass
                client.simSetVehiclePose(pose, True)
                print(f"  💥 Disturbance injected @ step {step}")
                client.moveToPositionAsync(target_pos.x_val, target_pos.y_val, target_pos.z_val, velocity=1)

            time.sleep(0.1)

        print(f"  ✅ Collected {len(dataset['obs'])} samples so far")

    if len(dataset['obs']) == 0:
        print("\n❌ ERROR: No data collected!")
        return None

    obs_array = np.array(dataset['obs'], dtype=np.float32)
    actions_array = np.array(dataset['actions'], dtype=np.float32)

    print("\n📊 Dataset (raw PWM) stats:")
    print(f"   Observations: {obs_array.shape}")
    print(f"   Actions:      {actions_array.shape}")
    print(f"   PWM mean:     {actions_array.mean(axis=0)}")
    print(f"   PWM std:      {actions_array.std(axis=0)}")
    print(f"   PWM range:    [{actions_array.min():.3f}, {actions_array.max():.3f}]")

    os.makedirs('data', exist_ok=True)
    np.save('data/hover_dataset.npy', {'obs': dataset['obs'], 'actions': dataset['actions']})
    np.save('data/normalization_params.npy', {'note': 'actions are raw PWM in [0,1]'})
    print("\n✅ Saved:")
    print("  • data/hover_dataset.npy")
    print("  • data/normalization_params.npy")

    return dataset

if __name__ == "__main__":
    collect_hover_data(num_episodes=10, steps_per_episode=200)
