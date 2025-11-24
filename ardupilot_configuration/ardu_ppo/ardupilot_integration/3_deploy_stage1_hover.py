"""
DEPLOY STAGE 1: HOVER POLICY
=============================
Deploys your trained Stage 1 behavioral cloning model on ArduPilot.

The model outputs velocity commands [vx, vy, vz] at 20Hz to maintain hover.

Usage:
    # SITL (safe testing)
    python 3_deploy_stage1_hover.py --connect 127.0.0.1:14550
    
    # Real hardware (use with caution!)
    python 3_deploy_stage1_hover.py --connect /dev/ttyUSB0 --duration 60
"""

import sys
sys.path.append('./utils')

from ardupilot_interface import ArduPilotInterface
from model_loader import ModelLoader, get_observation_from_vehicle
import numpy as np
import time
import argparse


def deploy_stage1_hover(connection_string, model_path, target_altitude=10.0, duration=120):
    """
    Deploy Stage 1 hover policy
    
    Args:
        connection_string: MAVLink connection
        model_path: Path to trained model
        target_altitude: Target hover altitude (m)
        duration: Test duration (seconds)
    """
    
    print("\n" + "="*70)
    print("🚁 DEPLOYING STAGE 1: HOVER POLICY")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Target altitude: {target_altitude}m")
    print(f"Duration: {duration}s")
    print(f"Control rate: 20Hz")
    print("="*70 + "\n")
    
    # Connect to drone
    print("[1/4] Connecting to drone...")
    client = ArduPilotInterface(connection_string)
    client.confirmConnection()
    
    # Load model
    print("\n[2/4] Loading trained model...")
    loader = ModelLoader()
    policy = loader.load_stage1(model_path)
    
    # Takeoff
    print("\n[3/4] Taking off...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync(target_altitude).join()
    time.sleep(2)
    
    print(f"✅ At {target_altitude}m altitude")
    
    # Deploy policy
    print("\n[4/4] Deploying hover policy...")
    print(f"{'='*70}")
    print("🎯 HOVER CONTROL ACTIVE")
    print(f"{'='*70}")
    print("Watch the drone maintain stable hover!")
    print("Press Ctrl+C to stop\n")
    
    # Control loop
    control_rate = 20  # Hz (same as training)
    dt = 1.0 / control_rate
    
    distances = []
    altitudes = []
    
    try:
        start_time = time.time()
        step = 0
        
        while (time.time() - start_time) < duration:
            loop_start = time.time()
            
            # Get observation (13 dimensions)
            obs = get_observation_from_vehicle(client)
            
            # Model predicts velocity commands
            action = policy.predict(obs)
            vx, vy, vz = action[0], action[1], action[2]
            
            # Send velocity command to drone
            client.moveByVelocityAsync(
                float(vx), float(vy), float(vz),
                duration=dt
            ).join()
            
            # Track metrics
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            alt = -pos.z_val  # Convert NED to altitude
            dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
            
            distances.append(dist_from_center)
            altitudes.append(alt)
            
            # Print status every 2 seconds
            if step % (control_rate * 2) == 0:
                elapsed = time.time() - start_time
                avg_dist = np.mean(distances[-control_rate*2:])
                avg_alt = np.mean(altitudes[-control_rate*2:])
                alt_error = abs(avg_alt - target_altitude)
                
                print(f"[{elapsed:6.1f}s] Altitude: {avg_alt:5.2f}m (error: {alt_error:4.2f}m) | "
                      f"Distance: {avg_dist:5.2f}m | "
                      f"Action: [{vx:5.2f}, {vy:5.2f}, {vz:5.2f}]")
            
            step += 1
            
            # Maintain control rate
            loop_time = time.time() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)
    
    except KeyboardInterrupt:
        print("\n⚠️  Control interrupted by user")
    
    # Statistics
    print("\n" + "="*70)
    print("📊 HOVER PERFORMANCE STATISTICS")
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
    print(f"  Mean distance from center: {mean_dist:.3f}m")
    print(f"  Max distance: {max_dist:.3f}m")
    
    print(f"\nControl:")
    print(f"  Total steps: {step}")
    print(f"  Control rate: {step / (time.time() - start_time):.1f} Hz")
    
    # Grade performance
    print(f"\n{'='*70}")
    if std_alt < 0.5 and mean_dist < 1.0:
        print("✅ EXCELLENT! Stable hover maintained")
    elif std_alt < 1.0 and mean_dist < 2.0:
        print("✅ GOOD! Hover is acceptable")
    else:
        print("⚠️  NEEDS IMPROVEMENT! Check model and gains")
    print("="*70 + "\n")
    
    # Land
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("✅ Stage 1 deployment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Stage 1 hover policy on ArduPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test in SITL for 2 minutes
  python 3_deploy_stage1_hover.py --connect 127.0.0.1:14550 --duration 120
  
  # Deploy on real hardware (30 seconds test)
  python 3_deploy_stage1_hover.py --connect /dev/ttyUSB0 --duration 30 --altitude 5
        """
    )
    
    parser.add_argument('--connect', type=str, default='127.0.0.1:14550',
                        help='Connection string (default: 127.0.0.1:14550)')
    parser.add_argument('--model', type=str, default='./models/hover_policy_best.pth',
                        help='Path to Stage 1 model')
    parser.add_argument('--altitude', type=float, default=10.0,
                        help='Target hover altitude in meters (default: 10m)')
    parser.add_argument('--duration', type=int, default=120,
                        help='Test duration in seconds (default: 120s)')
    
    args = parser.parse_args()
    
    deploy_stage1_hover(
        args.connect,
        args.model,
        args.altitude,
        args.duration
    )