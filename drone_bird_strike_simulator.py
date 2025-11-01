import airsim
import time
import random
import numpy as np

class BirdStrikeSimulator:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def simulate_bird_strike_realistic(self, 
                                       bird_mass=1.0,
                                       strike_velocity=15.0):
        """
        Actually working bird strike simulation.
        Uses velocity commands to create sudden disturbance.
        """
        
        print(f"⚠️  BIRD STRIKE! Mass:{bird_mass}kg, Velocity:{strike_velocity}m/s")
        
        # Get current state
        state = self.client.getMultirotorState()
        current_pos = state.kinematics_estimated.position
        current_vel = state.kinematics_estimated.linear_velocity
        
        # Calculate momentum transfer (Newton's laws)
        # Δv = (m_bird * v_bird) / m_drone
        # Assuming drone mass ~1.5kg (typical quadcopter)
        drone_mass = 1.5
        momentum = bird_mass * strike_velocity
        velocity_change = momentum / drone_mass
        
        # Random impact direction (birds hit from various angles)
        impact_angle_xy = random.uniform(0, 2*np.pi)
        impact_angle_z = random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees vertical
        
        # Calculate impact velocity vector
        impact_vx = velocity_change * np.cos(impact_angle_xy) * np.cos(impact_angle_z)
        impact_vy = velocity_change * np.sin(impact_angle_xy) * np.cos(impact_angle_z)
        impact_vz = velocity_change * np.sin(impact_angle_z)
        
        print(f"   Impact velocity change: vx={impact_vx:.2f}, vy={impact_vy:.2f}, vz={impact_vz:.2f} m/s")
        
        # METHOD 1: Use moveByVelocity (most direct)
        # Apply sudden velocity change for short duration
        strike_duration = 0.2  # seconds
        
        self.client.moveByVelocityAsync(
            impact_vx + current_vel.x_val,
            impact_vy + current_vel.y_val,
            impact_vz + current_vel.z_val,
            duration=strike_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        )
        
        time.sleep(strike_duration)
        
        # Add tumbling effect by commanding rapid position changes
        # This creates the "flip" behavior
        tumble_intensity = bird_mass / drone_mass
        
        for _ in range(3):  # Multiple tumbles
            # Random offset positions (simulates loss of control)
            offset_x = random.uniform(-tumble_intensity*2, tumble_intensity*2)
            offset_y = random.uniform(-tumble_intensity*2, tumble_intensity*2)
            offset_z = random.uniform(-tumble_intensity, tumble_intensity)
            
            target_pos = airsim.Vector3r(
                current_pos.x_val + offset_x,
                current_pos.y_val + offset_y,
                current_pos.z_val + offset_z
            )
            
            # Rapid movement creates instability
            self.client.moveToPositionAsync(
                target_pos.x_val,
                target_pos.y_val,
                target_pos.z_val,
                velocity=10,  # Fast movement
                timeout_sec=0.3
            )
            time.sleep(0.3)
        
        print("   Strike complete. Attempting hover recovery...")
        # Let flight controller stabilize
        time.sleep(1)
        
    def simulate_bird_strike_alternative(self, bird_mass=1.0):
        """
        Alternative method: Use angle commands to create flipping
        This is more dramatic and shows loss of control
        """
        print(f"⚠️  BIRD STRIKE (Flip method)! Mass:{bird_mass}kg")
        
        # Calculate tumble angles based on bird mass
        # Larger bird = more violent tumbling
        roll_change = random.uniform(-np.pi * bird_mass/2, np.pi * bird_mass/2)
        pitch_change = random.uniform(-np.pi * bird_mass/2, np.pi * bird_mass/2)
        yaw_change = random.uniform(-np.pi * bird_mass/3, np.pi * bird_mass/3)
        
        print(f"   Tumble angles: roll={np.degrees(roll_change):.1f}°, pitch={np.degrees(pitch_change):.1f}°")
        
        # Apply rapid angle changes (simulates being hit and flipping)
        duration = 0.5
        
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=roll_change / duration,  # rad/s
            pitch_rate=pitch_change / duration,
            yaw_rate=yaw_change / duration,
            throttle=0.6,  # Maintain some altitude
            duration=duration
        )
        
        time.sleep(duration)
        
        # Recovery phase - let it stabilize
        print("   Recovering from tumble...")
        self.client.moveByAngleRatesThrottleAsync(0, 0, 0, 0.59, 1.0)
        time.sleep(2)
    
    def simulate_eagle_attack(self):
        """Large eagle strike - most violent"""
        self.simulate_bird_strike_realistic(
            bird_mass=4.5,
            strike_velocity=20.0
        )
    
    def simulate_hawk_attack(self):
        """Medium hawk strike"""
        self.simulate_bird_strike_realistic(
            bird_mass=1.2,
            strike_velocity=18.0
        )
    
    def simulate_small_bird_strike(self):
        """Small bird like pigeon"""
        self.simulate_bird_strike_realistic(
            bird_mass=0.4,
            strike_velocity=12.0
        )
        
    def simulate_bird_strike_with_wind(self):
        """Most realistic - combines movement + wind gust"""
        # Sudden wind gust simulates bird impact
        gust_wind = airsim.Vector3r(
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-5, 5)
        )
    
        print(f"⚠️  BIRD STRIKE + WIND GUST!")
        self.client.simSetWind(gust_wind)
    
    # Also apply movement disturbance
    self.simulate_bird_strike_realistic(bird_mass=2.0, strike_velocity=15.0)
    
    # Remove wind after impact
    time.sleep(1)
    self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def demo_all_strikes(self):
        """Demonstrate different bird strikes"""
        print("🚁 Starting bird strike demonstration...")
        
        # Take off and hover
        self.client.takeoffAsync().join()
        print("Hovering at 10m...")
        self.client.moveToZAsync(-10, 2).join()
        time.sleep(3)
        
        # Test 1: Small bird
        print("\n=== TEST 1: Small Bird (Pigeon) ===")
        self.simulate_small_bird_strike()
        time.sleep(3)
        
        # Recover position
        self.client.moveToZAsync(-10, 2).join()
        time.sleep(3)
        
        # Test 2: Medium bird
        print("\n=== TEST 2: Medium Bird (Hawk) ===")
        self.simulate_hawk_attack()
        time.sleep(3)
        
        # Recover position
        self.client.moveToZAsync(-10, 2).join()
        time.sleep(3)
        
        # Test 3: Large bird with flip method
        print("\n=== TEST 3: Large Bird with Flip (Eagle) ===")
        self.simulate_bird_strike_alternative(bird_mass=4.5)
        time.sleep(3)
        
        print("\n✅ Demonstration complete. Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)

# Run it
if __name__ == "__main__":
    sim = BirdStrikeSimulator()
    
    # Quick test - single strike
    sim.client.takeoffAsync().join()
    print("Drone hovering, preparing for strike in 3 seconds...")
    time.sleep(3)
    
    # Try the working method
    sim.simulate_hawk_attack()
    
    time.sleep(5)
    sim.client.landAsync().join()
    
    # OR run full demo:
    # sim.demo_all_strikes()