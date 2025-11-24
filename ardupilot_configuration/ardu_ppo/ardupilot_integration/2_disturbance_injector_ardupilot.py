"""
DISTURBANCE INJECTOR FOR ARDUPILOT
===================================
Injects disturbances by sending velocity/attitude commands.

IMPORTANT: 
- Real hardware: Use with EXTREME CAUTION! Start with low intensities.
- SITL: Safe to test with any intensity.

Usage:
    python 2_disturbance_injector_ardupilot.py --connect 127.0.0.1:14550
"""

import sys
sys.path.append('./utils')

from ardupilot_interface import ArduPilotInterface
import numpy as np
import time
from enum import Enum
import argparse


class DisturbanceType(Enum):
    """Types of disturbances"""
    WIND_GUST = "wind_gust"
    FLIP = "flip"
    SPIN = "spin"
    DROP = "drop"


class DisturbanceInjectorArduPilot:
    """
    Injects disturbances for testing recovery (ArduPilot version)
    
    Similar to AirSim disturbance_injector.py but uses MAVLink commands
    """
    
    def __init__(self, client):
        self.client = client
        
        # Disturbance parameters (CONSERVATIVE for real hardware!)
        self.params = {
            'wind_gust': {
                'force_range': (1, 3),  # m/s (conservative for real hardware)
                'duration': 0.5,
            },
            'flip': {
                'velocity_range': (2, 4),  # m/s (conservative)
            },
            'spin': {
                'yaw_rate_range': (30, 60),  # degrees/sec (conservative)
            },
            'drop': {
                'velocity_range': (-2, -1),  # m/s downward
                'duration': 0.3,
            }
        }
    
    def inject_disturbance(self, disturbance_type, intensity=1.0):
        """
        Inject a disturbance
        
        Args:
            disturbance_type: DisturbanceType enum
            intensity: 0.0 to 2.0 multiplier (USE <1.0 FOR REAL HARDWARE!)
        
        Returns:
            dict with disturbance info
        """
        
        if disturbance_type == DisturbanceType.WIND_GUST:
            return self._apply_wind_gust(intensity)
        elif disturbance_type == DisturbanceType.FLIP:
            return self._apply_flip(intensity)
        elif disturbance_type == DisturbanceType.SPIN:
            return self._apply_spin(intensity)
        elif disturbance_type == DisturbanceType.DROP:
            return self._apply_drop(intensity)
        else:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")
    
    def _apply_wind_gust(self, intensity):
        """Simulate wind gust with velocity command"""
        vel_min, vel_max = self.params['wind_gust']['force_range']
        velocity = np.random.uniform(vel_min, vel_max) * intensity
        
        # Random direction (horizontal)
        angle = np.random.uniform(0, 2 * np.pi)
        vx = velocity * np.cos(angle)
        vy = velocity * np.sin(angle)
        vz = 0
        
        # Apply for duration
        duration = self.params['wind_gust']['duration']
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        
        return {
            'type': 'wind_gust',
            'velocity': [vx, vy, vz],
            'magnitude': velocity,
            'intensity': intensity
        }
    
    def _apply_flip(self, intensity):
        """Simulate flip with sudden pitch/roll velocity"""
        vel_min, vel_max = self.params['flip']['velocity_range']
        velocity = np.random.uniform(vel_min, vel_max) * intensity
        
        # Random direction
        direction = np.random.choice(['forward', 'backward', 'left', 'right'])
        
        if direction == 'forward':
            vx, vy = velocity, 0
        elif direction == 'backward':
            vx, vy = -velocity, 0
        elif direction == 'left':
            vx, vy = 0, -velocity
        else:  # right
            vx, vy = 0, velocity
        
        # Apply sudden velocity
        self.client.moveByVelocityAsync(vx, vy, 0, 0.5).join()
        
        return {
            'type': 'flip',
            'direction': direction,
            'velocity': velocity,
            'intensity': intensity
        }
    
    def _apply_spin(self, intensity):
        """Simulate spin with yaw rate"""
        rate_min, rate_max = self.params['spin']['yaw_rate_range']
        yaw_rate = np.random.uniform(rate_min, rate_max) * intensity
        
        # Random direction
        direction = np.random.choice([-1, 1])
        yaw_rate *= direction
        
        # Apply yaw rate (would need custom MAVLink message)
        # For now, use velocity in circular motion
        duration = 1.0
        for _ in range(int(duration / 0.05)):
            # Create circular motion
            angle = yaw_rate * 0.05 * (np.pi / 180)
            vx = 2 * np.sin(angle)
            vy = 2 * np.cos(angle)
            self.client.moveByVelocityAsync(vx, vy, 0, 0.05).join()
        
        return {
            'type': 'spin',
            'yaw_rate': yaw_rate,
            'direction': 'cw' if direction > 0 else 'ccw',
            'intensity': intensity
        }
    
    def _apply_drop(self, intensity):
        """Simulate sudden altitude drop"""
        vel_min, vel_max = self.params['drop']['velocity_range']
        velocity = np.random.uniform(vel_min, vel_max) * intensity
        
        duration = self.params['drop']['duration']
        self.client.moveByVelocityAsync(0, 0, -velocity, duration).join()
        
        return {
            'type': 'drop',
            'velocity': velocity,
            'intensity': intensity
        }


def test_disturbances(connection_string):
    """Test disturbance injector"""
    
    print("\n" + "="*70)
    print("🧪 TESTING DISTURBANCE INJECTOR (ARDUPILOT)")
    print("="*70)
    print("⚠️  WARNING: Only use in SITL or with extreme caution on real hardware!")
    print("="*70 + "\n")
    
    # Connect
    client = ArduPilotInterface(connection_string)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff
    client.takeoffAsync(10.0).join()
    time.sleep(2)
    
    # Create injector
    injector = DisturbanceInjectorArduPilot(client)
    
    # Test each disturbance type
    disturbances = [
        DisturbanceType.WIND_GUST,
        DisturbanceType.DROP,
        DisturbanceType.FLIP,
        DisturbanceType.SPIN
    ]
    
    for dist_type in disturbances:
        print(f"\n{'='*70}")
        print(f"Testing: {dist_type.value}")
        print(f"{'='*70}")
        
        # Apply disturbance
        info = injector.inject_disturbance(dist_type, intensity=0.5)
        
        print(f"✅ Applied {dist_type.value}")
        print(f"   Info: {info}")
        
        # Wait and stabilize
        print("   Waiting 5 seconds to stabilize...")
        time.sleep(5)
    
    # Land
    print("\n" + "="*70)
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect', type=str, default='127.0.0.1:14550',
                        help='Connection string')
    args = parser.parse_args()
    
    test_disturbances(args.connect)