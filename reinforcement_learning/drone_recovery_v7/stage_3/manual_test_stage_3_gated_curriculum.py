"""
MANUAL DISTURBANCE DEMONSTRATION SCRIPT
========================================
Interactive demo with full drone control!

Controls:
    W/A/S/D - Directional bird attacks
    F - Flip
    G - Spin (changed from R to avoid conflict)
    T - Drop
    
    UP ARROW    - Increase altitude (move up)
    DOWN ARROW  - Decrease altitude (move down)
    
    + or = - Increase intensity
    - - Decrease intensity
    
    SPACE - Apply disturbance
    R - RESET (restart from beginning - takeoff and hover)
    Q - Quit

Usage:
    python demo_manual_control.py --model ./models/stage3_checkpoints/gated_curriculum_policy.zip
"""

import airsim
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_flip_recovery_env_injector_gated import DroneFlipRecoveryEnv
import time
import sys
import os

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  Install: pip install pynput")
    sys.exit(1)

from disturbance_injector import DisturbanceInjector, DisturbanceType


class ManualDroneControl:
    """Interactive demonstration with full drone control"""
    
    def __init__(self, model_path, vecnorm_path):
        print("\n" + "="*70)
        print("🎮 MANUAL DRONE CONTROL & DISTURBANCE DEMO")
        print("="*70)
        print("Loading model and setting up environment...")
        
        # Create environment
        def make_env():
            env = DroneFlipRecoveryEnv(
                target_altitude=30.0,
                max_steps=1000,
                wind_strength=3.0,
                flip_prob=0.0,
                debug=False
            )
            return env
        
        self.env = DummyVecEnv([make_env])
        self.env = VecNormalize.load(vecnorm_path, self.env)
        self.env.training = False
        self.env.norm_reward = False
        
        self.model = PPO.load(model_path, device='cpu')
        
        # Access environment
        base_env = self.env.venv.envs[0]
        self.actual_env = base_env.env if hasattr(base_env, 'env') else base_env
        
        self.client = self.actual_env.client
        self.injector = self.actual_env.disturbance_injector
        
        # Demo state
        self.intensity = 1.0
        self.selected_disturbance = DisturbanceType.BIRD_ATTACK
        self.selected_direction = 'front'
        self.running = True
        self.obs = None
        self.episode_step = 0
        self.total_disturbances = 0
        self.recoveries = 0
        self.currently_recovering = False
        
        # Altitude control
        self.target_altitude = 30.0
        self.altitude_step = 2.0
        self.min_altitude = 5.0
        self.max_altitude = 50.0
        
        # Control flags
        self.pending_disturbance = False
        self.pending_reset = False
        
        print("   ✅ Model loaded successfully!")
        print("="*70 + "\n")
        
        self._print_controls()
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()
    
    def _print_controls(self):
        """Display keyboard controls"""
        print("🎮 KEYBOARD CONTROLS:")
        print("="*70)
        print("  📍 DISTURBANCE SELECTION:")
        print("     W/A/S/D - Front/Left/Back/Right attack")
        print("     F - Flip")
        print("     G - Spin (yaw rotation)")
        print("     T - Drop")
        print()
        print("  🚁 ALTITUDE CONTROL:")
        print("     UP ARROW    - Increase altitude (+2m)")
        print("     DOWN ARROW  - Decrease altitude (-2m)")
        print()
        print("  ⚡ INTENSITY CONTROL:")
        print("     + or = - Increase intensity (0.5x - 2.0x)")
        print("     - - Decrease intensity")
        print()
        print("  🎯 ACTIONS:")
        print("     SPACE - Apply selected disturbance")
        print("     R - RESET (takeoff from beginning)")
        print("     Q - Quit demonstration")
        print("="*70 + "\n")
    
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            # Character keys
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                # Disturbance selection
                if char == 'w':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'front'
                    print(f"\n📍 FRONT ATTACK | Intensity: {self.intensity:.2f}x")
                elif char == 'a':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'left'
                    print(f"\n📍 LEFT ATTACK | Intensity: {self.intensity:.2f}x")
                elif char == 's':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'back'
                    print(f"\n📍 BACK ATTACK | Intensity: {self.intensity:.2f}x")
                elif char == 'd':
                    self.selected_disturbance = DisturbanceType.BIRD_ATTACK
                    self.selected_direction = 'right'
                    print(f"\n📍 RIGHT ATTACK | Intensity: {self.intensity:.2f}x")
                elif char == 'f':
                    self.selected_disturbance = DisturbanceType.FLIP
                    print(f"\n🔄 FLIP | Intensity: {self.intensity:.2f}x")
                elif char == 'g':  # Changed from 'r' to 'g' for SPIN
                    self.selected_disturbance = DisturbanceType.SPIN
                    print(f"\n🌀 SPIN | Intensity: {self.intensity:.2f}x")
                elif char == 't':
                    self.selected_disturbance = DisturbanceType.DROP
                    print(f"\n⬇️  DROP | Intensity: {self.intensity:.2f}x")
                elif char == 'r':  # RESET only
                    self.pending_reset = True
                    print("\n🔄 RESET requested...")
                elif char == 'q':
                    print("\n🛑 Quitting...")
                    self.running = False
                elif char in ['+', '=']:
                    self.intensity = min(2.0, self.intensity + 0.1)
                    print(f"\n⬆️  Intensity: {self.intensity:.2f}x")
                elif char == '-':
                    self.intensity = max(0.5, self.intensity - 0.1)
                    print(f"\n⬇️  Intensity: {self.intensity:.2f}x")
            
            # Special keys
            if key == keyboard.Key.space:
                self.pending_disturbance = True
            elif key == keyboard.Key.up:
                self._change_altitude(+self.altitude_step)
            elif key == keyboard.Key.down:
                self._change_altitude(-self.altitude_step)
            
        except AttributeError:
            pass
    
    def _change_altitude(self, delta):
        """Change target altitude"""
        old_altitude = self.target_altitude
        self.target_altitude = np.clip(
            self.target_altitude + delta,
            self.min_altitude,
            self.max_altitude
        )
        
        if self.target_altitude != old_altitude:
            print(f"\n🚁 Target altitude: {old_altitude:.1f}m → {self.target_altitude:.1f}m")
            
            # Get current position
            drone_state = self.client.getMultirotorState()
            pos = drone_state.kinematics_estimated.position
            
            # Move to new altitude while maintaining x,y position
            self.client.moveToPositionAsync(
                pos.x_val,
                pos.y_val,
                -self.target_altitude,
                velocity=3.0
            )
            print(f"   Moving to {self.target_altitude:.1f}m...")
        else:
            if delta > 0:
                print(f"\n⚠️  Already at max altitude ({self.max_altitude:.1f}m)")
            else:
                print(f"\n⚠️  Already at min altitude ({self.min_altitude:.1f}m)")
    
    def _reset_drone(self):
        """FULL RESET: Start from beginning (reset, takeoff, hover)"""
        print("\n" + "="*70)
        print("🔄 RESETTING DRONE - FULL RESTART")
        print("="*70)
        
        # Step 1: Reset AirSim
        print("   [1/4] Resetting AirSim...")
        self.client.reset()
        time.sleep(0.5)
        
        # Step 2: Enable API control and arm
        print("   [2/4] Enabling API control...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.5)
        
        # Step 3: Takeoff
        print("   [3/4] Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        
        # Step 4: Move to hover position at target altitude
        print(f"   [4/4] Moving to hover position (0, 0, {self.target_altitude:.1f}m)...")
        start_x = np.random.uniform(-1, 1)
        start_y = np.random.uniform(-1, 1)
        self.client.moveToPositionAsync(
            start_x, start_y, -self.target_altitude, 5
        ).join()
        time.sleep(1.0)
        
        # Reset environment observation
        self.obs = self.env.reset()
        
        # Reset tracking variables
        self.episode_step = 0
        self.currently_recovering = False
        self.actual_env.disturbance_initiated = False
        self.actual_env.disturbance_recovered = False
        
        print("   ✅ Reset complete! Drone ready for disturbance.")
        print("="*70 + "\n")
    
    def _display_status(self):
        """Show current status with recovery percentage"""
        intensity_bar = "█" * int(self.intensity * 10)
        empty_bar = "░" * (20 - len(intensity_bar))
        
        # Get current position
        drone_state = self.client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        current_alt = -pos.z_val
        
        # Calculate recovery percentage
        recovery_percentage = (self.recoveries / self.total_disturbances * 100) if self.total_disturbances > 0 else 0
        
        recovery_str = "🔄 RECOVERING" if self.currently_recovering else "✅ STABLE"
        
        print(f"\r{recovery_str} | Alt: {current_alt:5.1f}m | "
              f"{self.selected_disturbance.value:12s} | "
              f"[{intensity_bar}{empty_bar}] {self.intensity:.1f}x | "
              f"Disturbances: {self.total_disturbances} | "
              f"Recovered: {self.recoveries} | "
              f"Success: {recovery_percentage:.0f}%", 
              end='', flush=True)
    
    def _apply_disturbance(self):
        """Apply the currently selected disturbance"""
        print(f"\n\n🐦 APPLYING DISTURBANCE!")
        print(f"   Type: {self.selected_disturbance.value}")
        print(f"   Intensity: {self.intensity:.2f}x")
        
        if self.selected_disturbance == DisturbanceType.BIRD_ATTACK:
            self._apply_directional_bird_attack()
        else:
            info = self.injector.inject_disturbance(
                self.selected_disturbance,
                intensity=self.intensity
            )
            print(f"   Details: {info}")
        
        self.total_disturbances += 1
        self.currently_recovering = True
        
        self.actual_env.disturbance_initiated = True
        self.actual_env.disturbance_recovered = False
        self.actual_env.disturbance_start_step = self.episode_step
        
        print(f"   ⏱️  Monitoring recovery...")
    
    def _apply_directional_bird_attack(self):
        """Apply bird attack with specific direction"""
        force = np.random.uniform(40, 80) * self.intensity
        
        directions = {
            'front': (force, 0, 0),
            'back': (-force, 0, 0),
            'left': (0, -force, 0),
            'right': (0, force, 0)
        }
        
        fx, fy, fz = directions[self.selected_direction]
        
        vx, vy, vz = fx * 0.1, fy * 0.1, fz * 0.1
        
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.15,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        
        angular_vel = np.random.uniform(360, 540) * self.intensity
        axis = np.random.choice(['roll', 'pitch'])
        angular_vel_rad = np.radians(angular_vel)
        
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=angular_vel_rad if axis == 'roll' else 0,
            pitch_rate=angular_vel_rad if axis == 'pitch' else 0,
            yaw_rate=0,
            throttle=0.59375,
            duration=0.8
        ).join()
        
        print(f"   Direction: {self.selected_direction}")
        print(f"   Force: {force:.1f}N")
        print(f"   Angular velocity: {angular_vel:.1f} deg/s ({axis})")
    
    def run(self):
        """Run the demonstration"""
        print("🚀 STARTING DEMONSTRATION")
        print("="*70)
        print("Initial drone setup...")
        
        # Initial reset to setup drone
        self._reset_drone()
        
        print(f"✅ Ready! Hovering at {self.target_altitude:.1f}m")
        print("\n💡 TIPS:")
        print("   - UP/DOWN arrows: Change altitude")
        print("   - R: Full reset (restart from takeoff)")
        print("   - W/A/S/D: Select attack direction")
        print("   - F/G/T: Flip/Spin/Drop")
        print("   - +/-: Adjust intensity")
        print("   - SPACE: Apply disturbance")
        print("   - Recovery % shown in real-time\n")
        
        # Main loop
        while self.running:
            try:
                # Handle pending reset
                if self.pending_reset:
                    self._reset_drone()
                    self.pending_reset = False
                
                # Handle pending disturbance
                if self.pending_disturbance:
                    self._apply_disturbance()
                    self.pending_disturbance = False
                
                # Run policy step
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, reward, done, info = self.env.step(action)
                self.episode_step += 1
                
                # Check recovery
                if self.currently_recovering:
                    if self.actual_env.disturbance_recovered:
                        recovery_time = self.actual_env.recovery_steps
                        self.recoveries += 1
                        self.currently_recovering = False
                        
                        # Calculate current recovery percentage
                        recovery_percentage = (self.recoveries / self.total_disturbances * 100)
                        
                        print(f"\n   ✅ RECOVERED in {recovery_time} steps ({recovery_time*0.05:.1f}s)!")
                        print(f"      Recovery rate: {recovery_percentage:.0f}% ({self.recoveries}/{self.total_disturbances})")
                
                # Update display
                self._display_status()
                
                # Auto-reset if crashed
                if done:
                    reason = info[0].get('reason', 'unknown') if isinstance(info, list) else info.get('reason', 'unknown')
                    if reason != 'timeout':
                        print(f"\n\n💥 Episode ended: {reason}")
                        print("🔄 Auto-resetting in 3 seconds...")
                        time.sleep(3)
                        self._reset_drone()
                
                time.sleep(0.05)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                self.running = False
        
        # Cleanup
        self.listener.stop()
        self.env.close()
        
        print("\n" + "="*70)
        print("📊 DEMONSTRATION SUMMARY")
        print("="*70)
        print(f"   Total disturbances: {self.total_disturbances}")
        print(f"   Successful recoveries: {self.recoveries}")
        if self.total_disturbances > 0:
            recovery_rate = (self.recoveries / self.total_disturbances) * 100
            print(f"   Recovery rate: {recovery_rate:.1f}%")
        print("="*70 + "\n")
        print("✅ Demonstration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_policy.zip',
                        help='Path to trained model')
    parser.add_argument('--vecnorm', type=str,
                        default='./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl',
                        help='Path to VecNormalize stats')
    
    args = parser.parse_args()
    
    if not KEYBOARD_AVAILABLE:
        print("❌ Install: pip install pynput")
        sys.exit(1)
    
    demo = ManualDroneControl(args.model, args.vecnorm)
    demo.run()