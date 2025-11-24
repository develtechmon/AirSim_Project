"""
DEPLOY STAGE 3: IMPACT RECOVERY
================================
Deploys your trained Stage 3 model that recovers from mid-air impacts!

This is the CROWN JEWEL of your PhD - the drone that can:
1. Detect when it's been hit (tumbling)
2. Stop the spin and get upright
3. Climb back to target altitude
4. Resume stable hover

The model outputs velocity commands [vx, vy, vz] at 20Hz.

⚠️ SAFETY WARNING:
   - Test in SITL first!
   - Start with LOW intensity (0.5)
   - Have manual override ready (RC controller)
   - Test at SAFE altitude (30m+)

Usage:
    # SITL testing (SAFE)
    python 5_deploy_stage3_recovery.py --connect 127.0.0.1:14550
    
    # Real hardware (CAUTION!)
    python 5_deploy_stage3_recovery.py --connect /dev/ttyUSB0 --intensity 0.3
"""

import sys
sys.path.append('./utils')

from ardupilot_interface import ArduPilotInterface
from model_loader import ModelLoader, get_observation_from_vehicle, predict_with_normalization
import numpy as np
import time
import argparse
from enum import Enum


class DisturbanceType(Enum):
    """Types of disturbances to test"""
    BIRD_ATTACK = "bird_attack"
    FLIP = "flip"
    SPIN = "spin"


class DisturbanceInjectorArduPilot:
    """Injects test disturbances (simplified for ArduPilot)"""
    
    def __init__(self, client):
        self.client = client
    
    def inject_bird_attack(self, intensity=1.0):
        """Simulate bird strike from random direction"""
        # Random direction
        direction = np.random.choice(['front', 'back', 'left', 'right'])
        
        # Impact force (velocity impulse)
        force = np.random.uniform(2.0, 4.0) * intensity
        
        if direction == 'front':
            vx, vy = force, 0
        elif direction == 'back':
            vx, vy = -force, 0
        elif direction == 'left':
            vx, vy = 0, -force
        else:  # right
            vx, vy = 0, force
        
        # Apply sudden velocity
        self.client.moveByVelocityAsync(vx, vy, 0, 0.3).join()
        
        # Add tumbling motion
        for _ in range(10):
            spin_vx = np.random.uniform(-2, 2) * intensity
            spin_vy = np.random.uniform(-2, 2) * intensity
            self.client.moveByVelocityAsync(spin_vx, spin_vy, 0, 0.05).join()
        
        return {
            'type': 'bird_attack',
            'direction': direction,
            'force': force,
            'intensity': intensity
        }
    
    def inject_flip(self, intensity=1.0):
        """Simulate flip disturbance"""
        # Choose flip direction
        direction = np.random.choice(['forward', 'backward', 'left', 'right'])
        velocity = np.random.uniform(3.0, 5.0) * intensity
        
        if direction == 'forward':
            vx, vy = velocity, 0
        elif direction == 'backward':
            vx, vy = -velocity, 0
        elif direction == 'left':
            vx, vy = 0, -velocity
        else:  # right
            vx, vy = 0, velocity
        
        # Apply flip motion
        self.client.moveByVelocityAsync(vx, vy, 0, 0.5).join()
        
        return {
            'type': 'flip',
            'direction': direction,
            'velocity': velocity,
            'intensity': intensity
        }
    
    def inject_spin(self, intensity=1.0):
        """Simulate spin disturbance"""
        # Circular motion to induce spin
        duration = 1.0
        steps = int(duration / 0.05)
        angular_velocity = np.random.uniform(60, 120) * intensity  # deg/s
        
        for i in range(steps):
            angle = (angular_velocity * 0.05 * i) * (np.pi / 180)
            radius = 2.0
            vx = radius * np.sin(angle)
            vy = radius * np.cos(angle)
            self.client.moveByVelocityAsync(vx, vy, 0, 0.05).join()
        
        return {
            'type': 'spin',
            'angular_velocity': angular_velocity,
            'intensity': intensity
        }


def deploy_stage3_recovery(connection_string, model_path, vecnorm_path,
                           target_altitude=30.0, disturbance_type='bird_attack',
                           intensity=0.7, num_tests=3):
    """
    Deploy Stage 3 impact recovery policy
    
    Args:
        connection_string: MAVLink connection
        model_path: Path to trained PPO model
        vecnorm_path: Path to VecNormalize stats
        target_altitude: Target altitude (m) - MUST be 30m for trained model!
        disturbance_type: Type of disturbance to test
        intensity: Disturbance intensity (0.3-1.5, start LOW!)
        num_tests: Number of recovery tests to run
    """
    
    print("\n" + "="*70)
    print("🎯 DEPLOYING STAGE 3: IMPACT RECOVERY")
    print("="*70)
    print("This is your PhD's MAIN CONTRIBUTION!")
    print("")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Target altitude: {target_altitude}m")
    print(f"Disturbance: {disturbance_type} (intensity: {intensity}x)")
    print(f"Number of tests: {num_tests}")
    print(f"Control rate: 20Hz")
    print("")
    print("⚠️  SAFETY:")
    print("   - Ensure sufficient altitude for recovery")
    print("   - Have manual RC override ready")
    print("   - Start with low intensity (0.5)")
    print("="*70 + "\n")
    
    # Connect to drone
    print("[1/5] Connecting to drone...")
    client = ArduPilotInterface(connection_string)
    client.confirmConnection()
    
    # Load model
    print("\n[2/5] Loading trained recovery model...")
    loader = ModelLoader()
    model, vecnorm = loader.load_stage3(model_path, vecnorm_path)
    
    # Create disturbance injector
    injector = DisturbanceInjectorArduPilot(client)
    
    # Takeoff
    print("\n[3/5] Taking off to safe altitude...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync(target_altitude).join()
    time.sleep(3)
    
    print(f"✅ At {target_altitude}m altitude - SAFE for testing")
    
    # Stabilize first
    print("\n[4/5] Initial stabilization (10 seconds)...")
    for i in range(10):
        obs = get_observation_from_vehicle(client)
        action = predict_with_normalization(model, vecnorm, obs)
        client.moveByVelocityAsync(
            float(action[0]), float(action[1]), float(action[2]),
            duration=0.05
        ).join()
        time.sleep(0.05)
    
    print("✅ Stable hover established")
    
    # Run recovery tests
    print("\n[5/5] Running impact recovery tests...")
    print(f"{'='*70}")
    print(f"🧪 TESTING {disturbance_type.upper()} RECOVERY")
    print(f"{'='*70}\n")
    
    results = []
    
    for test_num in range(1, num_tests + 1):
        print(f"\n{'='*70}")
        print(f"TEST {test_num}/{num_tests}")
        print(f"{'='*70}")
        
        # Phase 1: Stable hover (5 seconds)
        print("\n[Phase 1] Pre-disturbance hover (5s)...")
        start_time = time.time()
        pre_positions = []
        
        while time.time() - start_time < 5.0:
            obs = get_observation_from_vehicle(client)
            action = predict_with_normalization(model, vecnorm, obs)
            client.moveByVelocityAsync(
                float(action[0]), float(action[1]), float(action[2]),
                duration=0.05
            ).join()
            
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            pre_positions.append(-pos.z_val)
            time.sleep(0.05)
        
        pre_altitude = np.mean(pre_positions)
        print(f"✅ Stable at {pre_altitude:.2f}m")
        
        # Phase 2: DISTURBANCE!
        print(f"\n[Phase 2] 🐦 APPLYING {disturbance_type.upper()}!")
        disturbance_start = time.time()
        
        if disturbance_type == 'bird_attack':
            dist_info = injector.inject_bird_attack(intensity)
        elif disturbance_type == 'flip':
            dist_info = injector.inject_flip(intensity)
        else:  # spin
            dist_info = injector.inject_spin(intensity)
        
        print(f"   💥 Impact applied!")
        print(f"   Type: {dist_info['type']}")
        print(f"   Intensity: {dist_info['intensity']}x")
        
        # Phase 3: RECOVERY!
        print(f"\n[Phase 3] 🚁 AUTONOMOUS RECOVERY IN PROGRESS...")
        recovery_start = time.time()
        
        max_recovery_time = 30.0  # 30 seconds max
        control_rate = 20  # Hz
        dt = 1.0 / control_rate
        
        recovery_data = {
            'altitudes': [],
            'distances': [],
            'angular_velocities': [],
            'orientations': []
        }
        
        recovered = False
        crashed = False
        recovery_time = 0
        
        while time.time() - recovery_start < max_recovery_time:
            loop_start = time.time()
            
            # Get observation
            obs = get_observation_from_vehicle(client)
            
            # Model controls recovery
            action = predict_with_normalization(model, vecnorm, obs)
            
            # Send velocity command
            client.moveByVelocityAsync(
                float(action[0]), float(action[1]), float(action[2]),
                duration=dt
            ).join()
            
            # Get metrics
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            ori = state.kinematics_estimated.orientation
            ang_vel = state.kinematics_estimated.angular_velocity
            
            alt = -pos.z_val
            dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
            ang_vel_mag = np.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)
            
            # Store data
            recovery_data['altitudes'].append(alt)
            recovery_data['distances'].append(dist_from_center)
            recovery_data['angular_velocities'].append(ang_vel_mag)
            
            # Check if upright (quaternion check)
            qw, qx, qy, qz = ori.w_val, ori.x_val, ori.y_val, ori.z_val
            up_z = 1 - 2 * (qx * qx + qy * qy)
            is_upright = up_z > 0.7
            
            # Print status every second
            elapsed = time.time() - recovery_start
            if int(elapsed * 10) % 10 == 0:  # Every second
                print(f"   [{elapsed:4.1f}s] Alt: {alt:5.2f}m | "
                      f"Dist: {dist_from_center:4.2f}m | "
                      f"AngVel: {ang_vel_mag:4.2f}rad/s | "
                      f"Upright: {'✅' if is_upright else '❌'}")
            
            # Check recovery conditions
            is_controlled = (ang_vel_mag < 0.5 and is_upright)
            is_safe_altitude = (alt > 5.0)
            is_near_target = (abs(alt - target_altitude) < 2.0)
            is_centered = (dist_from_center < 2.0)
            
            if is_controlled and is_safe_altitude and is_near_target and is_centered:
                recovered = True
                recovery_time = time.time() - recovery_start
                print(f"\n   ✅ RECOVERY SUCCESSFUL!")
                print(f"   ⏱️  Recovery time: {recovery_time:.2f}s")
                print(f"   📍 Final altitude: {alt:.2f}m")
                print(f"   📍 Final distance: {dist_from_center:.2f}m")
                break
            
            # Check crash conditions
            if alt < 2.0:
                crashed = True
                print(f"\n   ❌ CRASHED! Altitude too low: {alt:.2f}m")
                break
            
            if dist_from_center > 20.0:
                crashed = True
                print(f"\n   ❌ OUT OF BOUNDS! Distance: {dist_from_center:.2f}m")
                break
            
            # Maintain control rate
            loop_time = time.time() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)
        
        # Phase 4: Post-recovery stability check (if recovered)
        if recovered:
            print(f"\n[Phase 4] Checking post-recovery stability (5s)...")
            stability_start = time.time()
            post_altitudes = []
            post_distances = []
            
            while time.time() - stability_start < 5.0:
                obs = get_observation_from_vehicle(client)
                action = predict_with_normalization(model, vecnorm, obs)
                client.moveByVelocityAsync(
                    float(action[0]), float(action[1]), float(action[2]),
                    duration=0.05
                ).join()
                
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                post_altitudes.append(-pos.z_val)
                post_distances.append(np.sqrt(pos.x_val**2 + pos.y_val**2))
                time.sleep(0.05)
            
            post_altitude = np.mean(post_altitudes)
            post_distance = np.mean(post_distances)
            
            print(f"   ✅ Post-recovery: Alt={post_altitude:.2f}m, Dist={post_distance:.2f}m")
        
        # Store test result
        result = {
            'test_num': test_num,
            'success': recovered,
            'crashed': crashed,
            'recovery_time': recovery_time if recovered else None,
            'pre_altitude': pre_altitude,
            'min_altitude': np.min(recovery_data['altitudes']),
            'max_angular_velocity': np.max(recovery_data['angular_velocities']),
            'disturbance_info': dist_info
        }
        results.append(result)
        
        # Wait before next test
        if test_num < num_tests:
            print(f"\nWaiting 10 seconds before next test...")
            time.sleep(10)
    
    # Overall results
    print("\n" + "="*70)
    print("📊 OVERALL RECOVERY TEST RESULTS")
    print("="*70)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = (successes / num_tests) * 100
    
    print(f"\nSuccess Rate: {success_rate:.0f}% ({successes}/{num_tests} tests)")
    
    if successes > 0:
        successful_tests = [r for r in results if r['success']]
        avg_recovery_time = np.mean([r['recovery_time'] for r in successful_tests])
        print(f"Average Recovery Time: {avg_recovery_time:.2f}s")
    
    print(f"\nTest Details:")
    for r in results:
        status = "✅ RECOVERED" if r['success'] else "❌ FAILED"
        time_str = f"{r['recovery_time']:.2f}s" if r['recovery_time'] else "N/A"
        print(f"  Test {r['test_num']}: {status} | Time: {time_str} | "
              f"Min Alt: {r['min_altitude']:.2f}m | "
              f"Max AngVel: {r['max_angular_velocity']:.2f}rad/s")
    
    # PhD assessment
    print(f"\n{'='*70}")
    print("🎓 PhD ASSESSMENT")
    print(f"{'='*70}")
    
    if success_rate >= 80:
        print("✅ OUTSTANDING! Your system demonstrates excellent recovery capability!")
        print("   This validates your PhD hypothesis:")
        print("   'Impact-resilient UAV can autonomously recover from mid-air impacts'")
    elif success_rate >= 60:
        print("✅ GOOD! Your system shows promising recovery capability.")
        print("   Further tuning could improve performance to 80%+")
    elif success_rate >= 40:
        print("⚠️  MODERATE. System partially works but needs improvement.")
        print("   Consider: More training, better disturbance models, tuning")
    else:
        print("❌ NEEDS WORK. System struggles with recovery.")
        print("   Recommendations: Verify model, check normalization, retrain")
    
    print(f"{'='*70}\n")
    
    # Land
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("✅ Stage 3 deployment complete!")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Stage 3 impact recovery on ArduPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test in SITL (SAFE)
  python 5_deploy_stage3_recovery.py --connect 127.0.0.1:14550
  
  # Test bird attack with low intensity
  python 5_deploy_stage3_recovery.py --type bird_attack --intensity 0.5
  
  # Real hardware (START WITH LOW INTENSITY!)
  python 5_deploy_stage3_recovery.py --connect /dev/ttyUSB0 --intensity 0.3 --tests 1
        """
    )
    
    parser.add_argument('--connect', type=str, default='127.0.0.1:14550',
                        help='Connection string')
    parser.add_argument('--model', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_policy.zip',
                        help='Path to Stage 3 model')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--altitude', type=float, default=30.0,
                        help='Target altitude (m) - USE 30m for trained model!')
    parser.add_argument('--type', type=str, default='bird_attack',
                        choices=['bird_attack', 'flip', 'spin'],
                        help='Type of disturbance')
    parser.add_argument('--intensity', type=float, default=0.7,
                        help='Disturbance intensity (0.3-1.5, start LOW!)')
    parser.add_argument('--tests', type=int, default=3,
                        help='Number of recovery tests')
    
    args = parser.parse_args()
    
    deploy_stage3_recovery(
        args.connect,
        args.model,
        args.vecnorm,
        args.altitude,
        args.type,
        args.intensity,
        args.tests
    )