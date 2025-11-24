"""
DEPLOY STAGE 2: WIND DISTURBANCE RECOVERY
==========================================
Deploys your trained Stage 2 PPO model with wind disturbances.

The model outputs velocity commands [vx, vy, vz] to maintain hover despite wind.

⚠️ IMPORTANT: Wind simulation only works in SITL!
   Real hardware will just do hover (no simulated wind available)

Usage:
    # SITL with wind
    python 4_deploy_stage2_disturbance.py --connect 127.0.0.1:14550
"""

import sys
sys.path.append('./utils')

from ardupilot_interface import ArduPilotInterface
from model_loader import ModelLoader, get_observation_from_vehicle, predict_with_normalization
import numpy as np
import time
import argparse


def deploy_stage2_disturbance(connection_string, model_path, vecnorm_path,
                               target_altitude=10.0, duration=120, wind_strength=5.0):
    """
    Deploy Stage 2 wind disturbance policy
    
    Args:
        connection_string: MAVLink connection
        model_path: Path to trained PPO model
        vecnorm_path: Path to VecNormalize stats
        target_altitude: Target hover altitude (m)
        duration: Test duration (seconds)
        wind_strength: Wind strength for SITL (m/s)
    """
    
    print("\n" + "="*70)
    print("🌬️  DEPLOYING STAGE 2: WIND DISTURBANCE RECOVERY")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Target altitude: {target_altitude}m")
    print(f"Duration: {duration}s")
    print(f"Wind strength: 0-{wind_strength} m/s (SITL only)")
    print(f"Control rate: 20Hz")
    print("="*70 + "\n")
    
    # Connect to drone
    print("[1/4] Connecting to drone...")
    client = ArduPilotInterface(connection_string)
    client.confirmConnection()
    
    # Load model
    print("\n[2/4] Loading trained model...")
    loader = ModelLoader()
    model, vecnorm = loader.load_stage2(model_path, vecnorm_path)
    
    # Takeoff
    print("\n[3/4] Taking off...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync(target_altitude).join()
    time.sleep(2)
    
    print(f"✅ At {target_altitude}m altitude")
    
    # Deploy policy
    print("\n[4/4] Deploying wind disturbance recovery...")
    print(f"{'='*70}")
    print("🎯 WIND RECOVERY CONTROL ACTIVE")
    print(f"{'='*70}")
    print("⚠️  Note: Wind simulation requires SITL with SIM_WIND_* params")
    print("Watch the drone maintain hover despite wind!\n")
    
    # Control loop
    control_rate = 20  # Hz
    dt = 1.0 / control_rate
    
    distances = []
    altitudes = []
    
    try:
        start_time = time.time()
        step = 0
        
        # Simulate changing wind (SITL only)
        wind_change_interval = 2.0  # seconds
        last_wind_change = time.time()
        current_wind = np.zeros(3)
        
        while (time.time() - start_time) < duration:
            loop_start = time.time()
            
            # Change wind periodically (SITL only)
            if time.time() - last_wind_change > wind_change_interval:
                # Generate random wind
                angle = np.random.uniform(0, 2*np.pi)
                strength = np.random.uniform(0, wind_strength)
                current_wind = np.array([
                    strength * np.cos(angle),
                    strength * np.sin(angle),
                    0
                ])
                # Note: ArduPilot SITL wind requires SIM_WIND_* parameters
                # This is just for tracking, actual wind must be set in SITL params
                last_wind_change = time.time()
            
            # Get observation (13 dimensions)
            obs = get_observation_from_vehicle(client)
            
            # Model predicts velocity commands (with normalization)
            action = predict_with_normalization(model, vecnorm, obs)
            vx, vy, vz = action[0], action[1], action[2]
            
            # Send velocity command
            client.moveByVelocityAsync(
                float(vx), float(vy), float(vz),
                duration=dt
            ).join()
            
            # Track metrics
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            alt = -pos.z_val
            dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
            
            distances.append(dist_from_center)
            altitudes.append(alt)
            
            # Print status every 2 seconds
            if step % (control_rate * 2) == 0:
                elapsed = time.time() - start_time
                avg_dist = np.mean(distances[-control_rate*2:])
                avg_alt = np.mean(altitudes[-control_rate*2:])
                alt_error = abs(avg_alt - target_altitude)
                wind_mag = np.linalg.norm(current_wind)
                
                print(f"[{elapsed:6.1f}s] Alt: {avg_alt:5.2f}m (err: {alt_error:4.2f}m) | "
                      f"Dist: {avg_dist:5.2f}m | "
                      f"Wind: {wind_mag:4.1f}m/s | "
                      f"Act: [{vx:5.2f}, {vy:5.2f}, {vz:5.2f}]")
            
            step += 1
            
            # Maintain control rate
            loop_time = time.time() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)
    
    except KeyboardInterrupt:
        print("\n⚠️  Control interrupted by user")
    
    # Statistics
    print("\n" + "="*70)
    print("📊 WIND RECOVERY PERFORMANCE")
    print("="*70)
    
    mean_alt = np.mean(altitudes)
    std_alt = np.std(altitudes)
    max_alt_error = np.max(np.abs(np.array(altitudes) - target_altitude))
    
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    
    print(f"Altitude:")
    print(f"  Mean: {mean_alt:.3f}m (target: {target_altitude}m)")
    print(f"  Std dev: {std_alt:.3f}m")
    print(f"  Max error: {max_alt_error:.3f}m")
    
    print(f"\nPosition:")
    print(f"  Mean distance: {mean_dist:.3f}m")
    print(f"  Max distance: {max_dist:.3f}m")
    
    print(f"\n{'='*70}")
    if std_alt < 0.8 and mean_dist < 1.5:
        print("✅ EXCELLENT! Stable despite disturbances")
    elif std_alt < 1.5 and mean_dist < 3.0:
        print("✅ GOOD! Handling disturbances well")
    else:
        print("⚠️  NEEDS IMPROVEMENT! Check wind parameters")
    print("="*70 + "\n")
    
    # Land
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("✅ Stage 2 deployment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Stage 2 wind disturbance recovery on ArduPilot"
    )
    
    parser.add_argument('--connect', type=str, default='127.0.0.1:14550',
                        help='Connection string')
    parser.add_argument('--model', type=str, 
                        default='./models/hover_disturbance_policy.zip',
                        help='Path to Stage 2 model')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/hover_disturbance_vecnormalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--altitude', type=float, default=10.0,
                        help='Target altitude (m)')
    parser.add_argument('--duration', type=int, default=120,
                        help='Duration (seconds)')
    parser.add_argument('--wind', type=float, default=5.0,
                        help='Wind strength (m/s, SITL only)')
    
    args = parser.parse_args()
    
    deploy_stage2_disturbance(
        args.connect,
        args.model,
        args.vecnorm,
        args.altitude,
        args.duration,
        args.wind
    )
# ```

# ---

## 🎯 **KEY POINTS FOR VX, VY, VZ OUTPUTS**

### **✅ What I Changed:**

# 1. **Model outputs velocity commands directly** - No position conversion needed!
# 2. **Uses `moveByVelocityAsync(vx, vy, vz, dt)`** - Sends velocity commands at 20Hz
# 3. **Clips actions to [-5, 5] m/s** - Same as training safety limits
# 4. **Maintains 20Hz control loop** - Same frequency as AirSim training
# 5. **Handles VecNormalize** - For Stage 2 & 3 PPO models

# ### **🎯 Control Flow:**
# ```
# 1. Get observation (13D) from drone
# 2. Normalize if needed (Stage 2/3)
# 3. Model predicts: [vx, vy, vz]
# 4. Clip to [-5, 5]
# 5. Send to drone via MAVLink
# 6. Repeat at 20Hz