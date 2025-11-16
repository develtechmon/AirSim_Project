"""
Disturbance Injector for Impact Recovery Training - FIXED VERSION
===================================================================
✅ FIXED: Bird attacks now accept direction parameter (front/left/back/right)
✅ Maintains backward compatibility (random if direction not specified)

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
    ✅ FIXED: Now supports directional bird attacks!
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
                'angular_impulse': (360, 540),   # degrees/sec - SEVERE IMPACT
            },
            'wind_gust': {
                'force_range': (20, 60),
                'duration': 0.3,
            },
            'flip': {
                'angular_velocity': (450, 720), # degrees/sec - EXTREME FLIP
                'axis': ['roll', 'pitch'],      # which axis to flip
            },
            'spin': {
                'angular_velocity': (540, 900), # degrees/sec - EXTREME SPIN
            },
            'drop': {
                'force_range': (-80, -40),      # Downward force
                'duration': 0.2,
            }
        }
    
    def inject_disturbance(self, disturbance_type, intensity=1.0, direction=None):
        """
        Inject a disturbance to the drone
        
        Args:
            disturbance_type: DisturbanceType enum
            intensity: 0.0 to 2.0 (multiplier for disturbance strength)
            direction: (OPTIONAL) For BIRD_ATTACK only - 'front', 'back', 'left', 'right'
                      If None, direction is random (backward compatible)
        
        Returns:
            dict: Information about the applied disturbance
        """
        
        if disturbance_type == DisturbanceType.COLLISION:
            return self._apply_collision(intensity)
        elif disturbance_type == DisturbanceType.BIRD_ATTACK:
            return self._apply_bird_attack(intensity, direction=direction)
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
    
    def _apply_bird_attack(self, intensity, direction=None):
        """
        Simulate bird strike (side impact with rotation)
        
        ✅ FIXED: Now accepts direction parameter!
        
        Args:
            intensity: Force multiplier
            direction: 'front', 'back', 'left', 'right' or None (random)
        """
        force_min, force_max = self.params['bird_attack']['force_range']
        force = np.random.uniform(force_min, force_max) * intensity
        
        # ✅ FIXED: Use specified direction or random if not provided
        if direction is None:
            direction = np.random.choice(['left', 'right', 'front', 'back'])
        
        # Validate direction
        if direction not in ['left', 'right', 'front', 'back']:
            print(f"⚠️ Invalid direction '{direction}', using random")
            direction = np.random.choice(['left', 'right', 'front', 'back'])
        
        # Set force based on direction
        fx, fy, fz = 0, 0, 0
        
        if direction == 'left':
            fy = -force  # Push from left (negative Y in NED)
        elif direction == 'right':
            fy = force   # Push from right (positive Y in NED)
        elif direction == 'front':
            fx = force   # Push from front (positive X in NED)
        else:  # back
            fx = -force  # Push from back (negative X in NED)
        
        # Apply force
        self._apply_force(fx, fy, fz, duration=self.params['bird_attack']['duration'])
        
        # Add angular impulse (causes rotation) - THIS IS KEY!
        ang_min, ang_max = self.params['bird_attack']['angular_impulse']
        angular_vel = np.random.uniform(ang_min, ang_max) * intensity
        
        # ✅ FIXED: Choose rotation axis AND direction based on impact direction
        # The drone should rotate AWAY from the impact
        if direction == 'front':
            # Hit from front → pitch backward (negative pitch in NED)
            axis = 'pitch'
            angular_vel = -angular_vel  # Negative = pitch back
        elif direction == 'back':
            # Hit from back → pitch forward (positive pitch in NED)
            axis = 'pitch'
            angular_vel = angular_vel  # Positive = pitch forward
        elif direction == 'left':
            # Hit from left → roll right (positive roll in NED)
            axis = 'roll'
            angular_vel = angular_vel  # Positive = roll right
        else:  # right
            # Hit from right → roll left (negative roll in NED)
            axis = 'roll'
            angular_vel = -angular_vel  # Negative = roll left
        
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
        
        # Duration of rotation application - LONGER for more dramatic effect!
        duration = 0.8  # seconds (was 0.5) - creates persistent rotation!
        
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