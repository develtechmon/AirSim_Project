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
    FLIP = "flip"                     # Sudden flip/roll
    SPIN = "spin"                     # Sudden yaw rotation
    DROP = "drop"                     # Sudden altitude drop


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
                'angular_impulse': (180, 360),   # degrees/sec (INCREASED!)
            },
            'wind_gust': {
                'force_range': (20, 60),
                'duration': 0.3,
            },
            'flip': {
                'angular_velocity': (270, 540), # degrees/sec (INCREASED!)
                'axis': ['roll', 'pitch'],      # which axis to flip
            },
            'spin': {
                'angular_velocity': (180, 360), # degrees/sec (INCREASED!)
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
        
        # Add angular impulse (causes rotation) - THIS IS KEY!
        ang_min, ang_max = self.params['bird_attack']['angular_impulse']
        angular_vel = np.random.uniform(ang_min, ang_max) * intensity
        axis = np.random.choice(['roll', 'pitch'])  # Random axis
        self._apply_angular_velocity(angular_vel, axis=axis)
        
        return {
            'type': 'bird_attack',
            'direction': direction,
            'force': [fx, fy, fz],
            'angular_velocity': angular_vel,
            'axis': axis,
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
    
    def _apply_force(self, fx, fy, fz, duration):
        """
        Apply force to drone (simulated by velocity command)
        """
        # Convert force to velocity impulse
        vx = fx * 0.1  # Scale factor
        vy = fy * 0.1
        vz = fz * 0.1
        
        # Apply velocity
        try:
            self.client.moveByVelocityAsync(
                vx, vy, vz,
                duration=duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
        except:
            pass
    
    def _apply_angular_velocity(self, angular_vel, axis='roll'):
        """
        Apply angular velocity (rotation) - THIS CREATES THE FLIP!
        
        Args:
            angular_vel: Angular velocity in degrees/sec
            axis: 'roll', 'pitch', or 'yaw'
        """
        # Convert to radians/sec
        angular_vel_rad = np.radians(angular_vel)
        
        # Duration of rotation application
        duration = 0.5  # seconds to apply rotation
        
        # Apply rotation using angle rates
        try:
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=angular_vel_rad if axis == 'roll' else 0,
                pitch_rate=angular_vel_rad if axis == 'pitch' else 0,
                yaw_rate=angular_vel_rad if axis == 'yaw' else 0,
                throttle=0.59375,  # Maintain hover
                duration=duration
            ).join()
        except:
            pass
    
    def get_random_disturbance(self):
        """Get a random disturbance type"""
        return np.random.choice(list(DisturbanceType))