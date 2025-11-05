"""
Disturbance Injector for Impact Recovery Training
Simulates various real-world disturbances that cause drone instability
"""

import airsim
import numpy as np
import time
from enum import Enum


class DisturbanceType(Enum):
    """Types of disturbances that can affect the drone"""
    COLLISION = "collision"           # Object collision
    BIRD_ATTACK = "bird_attack"       # Sudden side impact
    WIND_GUST = "wind_gust"          # Strong wind push
    ROTOR_FAILURE = "rotor_failure"   # Simulated motor failure
    FLIP = "flip"                     # Sudden flip/roll
    SPIN = "spin"                     # Sudden yaw rotation
    DROP = "drop"                     # Sudden altitude drop
    COMBINED = "combined"             # Multiple simultaneous disturbances


class DisturbanceInjector:
    """
    Injects various disturbances to simulate real-world impacts
    """
    
    def __init__(self, client):
        self.client = client
        
        # Disturbance parameters (tunable)
        self.params = {
            'collision': {
                'force_range': (50, 150),      # Newtons
                'duration': 0.1,                # seconds
            },
            'bird_attack': {
                'force_range': (30, 80),
                'duration': 0.15,
                'angular_impulse': (10, 30),   # degrees/sec
            },
            'wind_gust': {
                'force_range': (20, 60),
                'duration': 0.3,
            },
            'rotor_failure': {
                'duration': 0.5,
                'severity': 0.7,                # 70% thrust reduction
            },
            'flip': {
                'angular_velocity': (180, 360), # degrees/sec
                'axis': ['roll', 'pitch'],      # which axis to flip
            },
            'spin': {
                'angular_velocity': (90, 270),
            },
            'drop': {
                'force_range': (-80, -40),      # Downward force
                'duration': 0.2,
            }
        }
    
    def inject_disturbance(self, disturbance_type, intensity=1.0):
        """
        Inject a disturbance to the drone
        
        Args:
            disturbance_type: DisturbanceType enum
            intensity: 0.0 to 2.0 (multiplier for disturbance strength)
        
        Returns:
            dict: Information about the applied disturbance
        """
        
        if disturbance_type == DisturbanceType.COLLISION:
            return self._apply_collision(intensity)
        elif disturbance_type == DisturbanceType.BIRD_ATTACK:
            return self._apply_bird_attack(intensity)
        elif disturbance_type == DisturbanceType.WIND_GUST:
            return self._apply_wind_gust(intensity)
        elif disturbance_type == DisturbanceType.FLIP:
            return self._apply_flip(intensity)
        elif disturbance_type == DisturbanceType.SPIN:
            return self._apply_spin(intensity)
        elif disturbance_type == DisturbanceType.DROP:
            return self._apply_drop(intensity)
        elif disturbance_type == DisturbanceType.COMBINED:
            return self._apply_combined(intensity)
        else:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")
    
    def _apply_collision(self, intensity):
        """Simulate frontal collision impact"""
        force_min, force_max = self.params['collision']['force_range']
        force = np.random.uniform(force_min, force_max) * intensity
        
        # Random direction (mostly horizontal)
        angle = np.random.uniform(0, 2 * np.pi)
        fx = force * np.cos(angle)
        fy = force * np.sin(angle)
        fz = np.random.uniform(-force * 0.3, force * 0.3)
        
        # Apply impulse force
        self._apply_force(fx, fy, fz, duration=self.params['collision']['duration'])
        
        return {
            'type': 'collision',
            'force': [fx, fy, fz],
            'magnitude': force,
            'intensity': intensity
        }
    
    def _apply_bird_attack(self, intensity):
        """Simulate bird strike (side impact with rotation)"""
        force_min, force_max = self.params['bird_attack']['force_range']
        force = np.random.uniform(force_min, force_max) * intensity
        
        # Side impact
        direction = np.random.choice(['left', 'right', 'front', 'back'])
        fx, fy, fz = 0, 0, 0
        
        if direction == 'left':
            fy = -force
        elif direction == 'right':
            fy = force
        elif direction == 'front':
            fx = force
        else:  # back
            fx = -force
        
        # Apply force
        self._apply_force(fx, fy, fz, duration=self.params['bird_attack']['duration'])
        
        # Add angular impulse (causes rotation)
        ang_min, ang_max = self.params['bird_attack']['angular_impulse']
        angular_vel = np.random.uniform(ang_min, ang_max) * intensity
        self._apply_angular_velocity(angular_vel, axis='roll')
        
        return {
            'type': 'bird_attack',
            'direction': direction,
            'force': [fx, fy, fz],
            'angular_velocity': angular_vel,
            'intensity': intensity
        }
    
    def _apply_wind_gust(self, intensity):
        """Simulate strong wind gust"""
        force_min, force_max = self.params['wind_gust']['force_range']
        force = np.random.uniform(force_min, force_max) * intensity
        
        # Wind direction (horizontal)
        angle = np.random.uniform(0, 2 * np.pi)
        fx = force * np.cos(angle)
        fy = force * np.sin(angle)
        fz = np.random.uniform(-force * 0.2, force * 0.2)
        
        # Longer duration than collision
        self._apply_force(fx, fy, fz, duration=self.params['wind_gust']['duration'])
        
        return {
            'type': 'wind_gust',
            'force': [fx, fy, fz],
            'magnitude': force,
            'intensity': intensity
        }
    
    def _apply_flip(self, intensity):
        """Simulate sudden flip/roll"""
        vel_min, vel_max = self.params['flip']['angular_velocity']
        angular_vel = np.random.uniform(vel_min, vel_max) * intensity
        
        # Randomly choose roll or pitch
        axis = np.random.choice(self.params['flip']['axis'])
        
        self._apply_angular_velocity(angular_vel, axis=axis)
        
        return {
            'type': 'flip',
            'axis': axis,
            'angular_velocity': angular_vel,
            'intensity': intensity
        }
    
    def _apply_spin(self, intensity):
        """Simulate sudden yaw spin"""
        vel_min, vel_max = self.params['spin']['angular_velocity']
        angular_vel = np.random.uniform(vel_min, vel_max) * intensity
        
        # Random direction
        direction = np.random.choice([-1, 1])
        angular_vel *= direction
        
        self._apply_angular_velocity(angular_vel, axis='yaw')
        
        return {
            'type': 'spin',
            'angular_velocity': angular_vel,
            'direction': 'clockwise' if direction > 0 else 'counterclockwise',
            'intensity': intensity
        }
    
    def _apply_drop(self, intensity):
        """Simulate sudden altitude drop"""
        force_min, force_max = self.params['drop']['force_range']
        force = np.random.uniform(force_min, force_max) * intensity
        
        self._apply_force(0, 0, force, duration=self.params['drop']['duration'])
        
        return {
            'type': 'drop',
            'force': force,
            'intensity': intensity
        }
    
    def _apply_combined(self, intensity):
        """Apply multiple disturbances simultaneously"""
        # Randomly select 2-3 disturbances
        num_disturbances = np.random.randint(2, 4)
        available = [DisturbanceType.COLLISION, DisturbanceType.BIRD_ATTACK,
                    DisturbanceType.WIND_GUST, DisturbanceType.FLIP,
                    DisturbanceType.SPIN, DisturbanceType.DROP]
        
        selected = np.random.choice(available, size=num_disturbances, replace=False)
        
        results = []
        for dist_type in selected:
            result = self.inject_disturbance(dist_type, intensity * 0.7)  # Reduced intensity
            results.append(result)
        
        return {
            'type': 'combined',
            'disturbances': results,
            'intensity': intensity
        }
    
    def _apply_force(self, fx, fy, fz, duration):
        """
        Apply force to drone (simulated by velocity command)
        
        Note: AirSim doesn't have direct force API, so we simulate
        by applying velocity in the force direction
        """
        # Convert force to velocity impulse (simplified physics)
        # F = ma, assuming m=1kg for simplicity
        vx = fx * 0.1  # Scale factor
        vy = fy * 0.1
        vz = fz * 0.1
        
        # Apply velocity
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
    
    def _apply_angular_velocity(self, angular_vel, axis='roll'):
        """
        Apply angular velocity (rotation)
        
        Args:
            angular_vel: Angular velocity in degrees/sec
            axis: 'roll', 'pitch', or 'yaw'
        """
        # Get current state
        state = self.client.getMultirotorState()
        current_orientation = state.kinematics_estimated.orientation
        
        # Convert to Euler angles
        roll, pitch, yaw = airsim.to_eularian_angles(current_orientation)
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        # Calculate target angle based on axis
        duration = 0.3  # seconds to apply rotation
        
        if axis == 'roll':
            target_roll = roll_deg + angular_vel * duration
            target_pitch = pitch_deg
            target_yaw = yaw_deg
        elif axis == 'pitch':
            target_roll = roll_deg
            target_pitch = pitch_deg + angular_vel * duration
            target_yaw = yaw_deg
        else:  # yaw
            target_roll = roll_deg
            target_pitch = pitch_deg
            target_yaw = yaw_deg + angular_vel * duration
        
        # Apply rotation by moving to target orientation
        # Note: This is approximate, real implementation might need custom Unreal Engine physics
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=angular_vel if axis == 'roll' else 0,
            pitch_rate=angular_vel if axis == 'pitch' else 0,
            yaw_rate=angular_vel if axis == 'yaw' else 0,
            throttle=0.59375,  # Maintain hover
            duration=duration
        ).join()
    
    def get_random_disturbance(self):
        """Get a random disturbance type"""
        return np.random.choice(list(DisturbanceType))
    
    def get_curriculum_disturbance(self, training_progress):
        """
        Get disturbance based on curriculum learning
        
        Args:
            training_progress: 0.0 to 1.0 (percentage of training completed)
        
        Returns:
            (DisturbanceType, intensity)
        """
        if training_progress < 0.3:
            # Easy: Simple disturbances, low intensity
            disturbance = np.random.choice([
                DisturbanceType.WIND_GUST,
                DisturbanceType.DROP
            ])
            intensity = np.random.uniform(0.3, 0.6)
        
        elif training_progress < 0.6:
            # Medium: More variety, medium intensity
            disturbance = np.random.choice([
                DisturbanceType.COLLISION,
                DisturbanceType.FLIP,
                DisturbanceType.SPIN,
                DisturbanceType.WIND_GUST
            ])
            intensity = np.random.uniform(0.5, 1.0)
        
        else:
            # Hard: All disturbances, high intensity
            disturbance = self.get_random_disturbance()
            intensity = np.random.uniform(0.8, 1.5)
        
        return disturbance, intensity


# ==========================================
# STANDALONE TESTING
# ==========================================
if __name__ == "__main__":
    print("="*70)
    print("🎯 DISTURBANCE INJECTOR TEST")
    print("="*70)
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Take off
    print("\n🚁 Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Hover at position
    client.moveToPositionAsync(0, 0, -10, 5).join()
    print("✓ Hovering at (0, 0, -10)")
    time.sleep(2)
    
    # Create injector
    injector = DisturbanceInjector(client)
    
    # Test each disturbance type
    disturbances = [
        DisturbanceType.COLLISION,
        DisturbanceType.BIRD_ATTACK,
        DisturbanceType.WIND_GUST,
        DisturbanceType.FLIP,
        DisturbanceType.SPIN,
        DisturbanceType.DROP
    ]
    
    for dist_type in disturbances:
        print(f"\n{'='*70}")
        print(f"Testing: {dist_type.value.upper()}")
        print(f"{'='*70}")
        
        # Apply disturbance
        result = injector.inject_disturbance(dist_type, intensity=1.0)
        print(f"Applied: {result}")
        
        # Wait and observe
        time.sleep(3)
        
        # Return to hover position
        print("Returning to hover position...")
        client.moveToPositionAsync(0, 0, -10, 3).join()
        time.sleep(2)
    
    # Land
    print("\n🛬 Landing...")
    client.landAsync().join()
    
    # Cleanup
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✅ Test complete!")