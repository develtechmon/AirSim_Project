import airsim
import time
import random
import numpy as np

class ViolentBirdStrike:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Disable stabilization for more realistic tumbling
        # This makes the drone more vulnerable to disturbances
        
    def create_extreme_tumble(self, bird_mass=3.0):
        """
        Creates VIOLENT tumbling that looks like real bird strike.
        Uses multiple simultaneous disturbances.
        """
        print(f"\n💥💥💥 VIOLENT BIRD STRIKE - {bird_mass}kg eagle! 💥💥💥")
        
        # Step 1: Massive sudden wind gust (simulates impact force)
        # Research shows birds create localized "wind burst" on impact
        impact_direction = random.uniform(0, 2*np.pi)
        gust_magnitude = bird_mass * 15  # Huge wind proportional to bird mass
        
        gust_x = gust_magnitude * np.cos(impact_direction)
        gust_y = gust_magnitude * np.sin(impact_direction)
        gust_z = random.uniform(-bird_mass*3, bird_mass*2)  # Vertical component
        
        print(f"⚡ Applying wind gust: {gust_x:.1f}, {gust_y:.1f}, {gust_z:.1f} m/s")
        self.client.simSetWind(airsim.Vector3r(gust_x, gust_y, gust_z))
        
        # Step 2: Immediately command VIOLENT angular rates (the flip!)
        # This is the key - use EXTREME values
        roll_rate = random.uniform(-10, 10)   # rad/s - EXTREME
        pitch_rate = random.uniform(-10, 10)  # rad/s - EXTREME  
        yaw_rate = random.uniform(-5, 5)      # rad/s
        
        print(f"🌀 Tumbling: roll={roll_rate:.1f}, pitch={pitch_rate:.1f} rad/s")
        print(f"   (That's {np.degrees(roll_rate):.0f}°/s roll, {np.degrees(pitch_rate):.0f}°/s pitch!)")
        
        # Apply violent tumbling
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=roll_rate,
            pitch_rate=pitch_rate, 
            yaw_rate=yaw_rate,
            throttle=0.4,  # Low throttle = less stability
            duration=0.8
        ).join()
        
        # Step 3: Add chaotic velocity during tumble
        chaos_vx = random.uniform(-15, 15)
        chaos_vy = random.uniform(-15, 15)
        chaos_vz = random.uniform(-5, 10)
        
        self.client.moveByVelocityAsync(
            chaos_vx, chaos_vy, chaos_vz,
            duration=0.5,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        )
        
        time.sleep(0.5)
        
        # Step 4: Continue tumbling with different rates (looks more chaotic)
        roll_rate2 = random.uniform(-8, 8)
        pitch_rate2 = random.uniform(-8, 8)
        
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=roll_rate2,
            pitch_rate=pitch_rate2,
            yaw_rate=random.uniform(-3, 3),
            throttle=0.5,
            duration=0.6
        ).join()
        
        # Step 5: Gradually reduce wind (bird tumbles away)
        print("🌪️  Reducing wind gust...")
        self.client.simSetWind(airsim.Vector3r(gust_x*0.5, gust_y*0.5, gust_z*0.5))
        time.sleep(0.3)
        
        self.client.simSetWind(airsim.Vector3r(gust_x*0.2, gust_y*0.2, gust_z*0.2))
        time.sleep(0.3)
        
        # Step 6: Remove wind, let drone try to recover
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        print("✋ Strike over. Drone attempting recovery...")
        
        # Give flight controller a chance to stabilize
        time.sleep(2)
    
    def create_propeller_strike(self):
        """
        Simulates bird hitting propeller - causes instant severe yaw spin
        """
        print("\n⚠️  PROPELLER STRIKE - Asymmetric thrust loss!")
        
        # Sudden massive yaw with wind
        wind_side_impact = airsim.Vector3r(
            random.uniform(-25, 25),
            random.uniform(-25, 25),
            -5
        )
        self.client.simSetWind(wind_side_impact)
        
        # Massive yaw spin (one prop damaged)
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-5, 5),
            pitch_rate=random.uniform(-5, 5),
            yaw_rate=random.uniform(-15, 15),  # EXTREME yaw
            throttle=0.3,
            duration=1.0
        ).join()
        
        # Remove wind
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        time.sleep(2)
    
    def create_multi_bird_mob(self, num_birds=5):
        """
        Multiple birds attacking - continuous chaotic disturbances
        """
        print(f"\n🦅🦅🦅 BIRD MOB - {num_birds} birds swarming! 🦅🦅🦅")
        
        for i in range(num_birds):
            print(f"\n🐦 Bird {i+1} attacking...")
            
            # Each bird creates a disturbance
            wind = airsim.Vector3r(
                random.uniform(-15, 15),
                random.uniform(-15, 15),
                random.uniform(-5, 5)
            )
            self.client.simSetWind(wind)
            
            # Quick tumble
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-6, 6),
                pitch_rate=random.uniform(-6, 6),
                yaw_rate=random.uniform(-3, 3),
                throttle=0.5,
                duration=0.4
            )
            
            # Short pause between attacks
            time.sleep(0.3)
        
        # Clear wind
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        print("\n✅ Mob dispersed")
        time.sleep(2)
    
    def extreme_eagle_attack(self):
        """
        The most violent strike - large eagle at high speed
        This WILL make the drone flip multiple times
        """
        print("\n🦅💥 EXTREME EAGLE ATTACK 💥🦅")
        print("Massive 5kg eagle at 25 m/s - BRACE FOR IMPACT!")
        
        # Extreme wind burst
        angle = random.uniform(0, 2*np.pi)
        magnitude = 75  # HUGE wind
        
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            -15
        )
        
        print(f"💨 Hurricane-force gust: {wind.x_val:.0f}, {wind.y_val:.0f}, {wind.z_val:.0f} m/s")
        self.client.simSetWind(wind)
        
        # EXTREME tumbling
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=12.0,   # 687°/sec - INSANE
            pitch_rate=12.0,
            yaw_rate=8.0,
            throttle=0.2,     # Minimal throttle
            duration=1.2
        ).join()
        
        # Continue chaos
        self.client.moveByVelocityAsync(
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-10, 5),
            duration=0.8,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        )
        
        time.sleep(0.8)
        
        # More tumbling
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=-10.0,
            pitch_rate=-10.0,
            yaw_rate=-6.0,
            throttle=0.3,
            duration=1.0
        ).join()
        
        # Gradually reduce wind
        for i in range(5):
            factor = 1.0 - (i * 0.2)
            self.client.simSetWind(airsim.Vector3r(
                wind.x_val * factor,
                wind.y_val * factor,
                wind.z_val * factor
            ))
            time.sleep(0.2)
        
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        print("\n🏥 Strike complete. Checking if drone survived...")
        time.sleep(3)
    
    def test_recovery_ability(self):
        """
        Test if your control algorithm can recover from strike
        """
        print("\n🧪 RECOVERY TEST SEQUENCE 🧪")
        
        # Take off
        self.client.takeoffAsync().join()
        print("Hovering at 20m...")
        self.client.moveToZAsync(-20, 3).join()
        time.sleep(2)
        
        # Get initial position
        state = self.client.getMultirotorState()
        initial_pos = state.kinematics_estimated.position
        
        print(f"Initial position: ({initial_pos.x_val:.1f}, {initial_pos.y_val:.1f}, {initial_pos.z_val:.1f})")
        
        # STRIKE!
        self.create_extreme_tumble(bird_mass=3.0)
        
        # Check final position
        time.sleep(3)
        state = self.client.getMultirotorState()
        final_pos = state.kinematics_estimated.position
        
        # Calculate displacement
        displacement = np.sqrt(
            (final_pos.x_val - initial_pos.x_val)**2 +
            (final_pos.y_val - initial_pos.y_val)**2 +
            (final_pos.z_val - initial_pos.z_val)**2
        )
        
        altitude_loss = final_pos.z_val - initial_pos.z_val
        
        print(f"\n📊 RECOVERY ANALYSIS:")
        print(f"   Total displacement: {displacement:.2f}m")
        print(f"   Altitude loss: {altitude_loss:.2f}m")
        print(f"   Final position: ({final_pos.x_val:.1f}, {final_pos.y_val:.1f}, {final_pos.z_val:.1f})")
        
        # Check if still flying
        if abs(final_pos.z_val) < 2:
            print("   ❌ CRASHED - Drone hit ground")
        elif abs(final_pos.z_val) > 50:
            print("   ❌ LOST CONTROL - Drone flew away")
        else:
            print("   ✅ SURVIVED - Drone recovered!")
        
        time.sleep(2)
        
        try:
            self.client.landAsync().join()
        except:
            print("   ⚠️  Landing failed - drone may have crashed")
    
    def demo_all_attack_types(self):
        """
        Full demonstration of all bird strike types
        """
        print("=" * 60)
        print("🦅 COMPLETE BIRD STRIKE DEMONSTRATION 🦅")
        print("=" * 60)
        
        # Takeoff
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-15, 3).join()
        
        # Test 1: Small bird
        print("\n" + "="*60)
        input("Press ENTER for TEST 1: Small Bird Strike (1kg)...")
        self.create_extreme_tumble(bird_mass=1.0)
        self.client.moveToZAsync(-15, 2).join()
        time.sleep(2)
        
        # Test 2: Propeller strike
        print("\n" + "="*60)
        input("Press ENTER for TEST 2: Propeller Strike...")
        self.create_propeller_strike()
        self.client.moveToZAsync(-15, 2).join()
        time.sleep(2)
        
        # Test 3: Mob attack
        print("\n" + "="*60)
        input("Press ENTER for TEST 3: Bird Mob (5 birds)...")
        self.create_multi_bird_mob(5)
        self.client.moveToZAsync(-15, 2).join()
        time.sleep(2)
        
        # Test 4: EXTREME eagle
        print("\n" + "="*60)
        input("Press ENTER for TEST 4: EXTREME EAGLE ATTACK (WARNING: VIOLENT!)...")
        self.extreme_eagle_attack()
        
        time.sleep(3)
        
        print("\n✅ Demonstration complete!")
        self.client.landAsync().join()


# ============= RUN IT =============
if __name__ == "__main__":
    print("🚁 Bird Strike Simulator - Realistic Violence Mode 🦅")
    print("WARNING: Drone will flip violently!\n")
    
    sim = ViolentBirdStrike()
    
    # Option 1: Quick extreme test
    sim.client.takeoffAsync().join()
    sim.client.moveToZAsync(-20, 3).join()
    print("\nDrone stable. Strike in 3 seconds...")
    time.sleep(3)
    
    sim.extreme_eagle_attack()
    
    time.sleep(5)
    try:
        sim.client.landAsync().join()
    except:
        print("Drone may have crashed - that's realistic!")
    
    # Option 2: Full demo
    # sim.demo_all_attack_types()
    
    # Option 3: Recovery test
    # sim.test_recovery_ability()