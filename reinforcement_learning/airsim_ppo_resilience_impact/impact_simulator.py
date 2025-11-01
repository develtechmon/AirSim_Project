"""
Impact Simulator for AirSim
============================

Simulates different types of physical impacts on a drone:
1. Sharp Collision (bird strike, wall hit)
2. Sustained Force (wind gust, downdraft)
3. Rotational Disturbance (asymmetric collision, prop failure)
4. Free-fall (complete thrust loss)

Each impact type has distinct IMU signatures based on literature:
- Sharp Collision: High jerk (>30 m/s³), brief (<0.2s), directional
- Sustained Force: Low jerk (<10 m/s³), long (>1s), continuous
- Rotational: High angular accel (>100°/s²), torque imbalance
- Free-fall: Sudden vertical accel drop (<5 m/s²)

References:
- MDPI Drones 2025: Collision detection with IMU
- Sensors 2021: Fault detection with FFT+WPD
"""

import airsim
import numpy as np
import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict


class ImpactType(Enum):
    """
    Impact classification based on MEASURABLE physical characteristics,
    not source identification (we can't tell bird from wall).
    """
    SHARP_COLLISION = "sharp_collision"      # Brief, high jerk (bird, wall)
    SUSTAINED_FORCE = "sustained_force"      # Continuous push (wind, downdraft)
    ROTATIONAL = "rotational"                # Spinning (asymmetric hit)
    FREE_FALL = "free_fall"                  # Thrust loss
    

@dataclass
class ImpactParameters:
    """Parameters defining an impact event"""
    impact_type: ImpactType
    magnitude: float  # Impact strength (1.0 = light, 3.0 = severe)
    direction: np.ndarray  # 3D direction vector
    duration: float  # Duration in seconds
    angular_component: float = 0.0  # Rotational component (rad/s)


class ImpactSimulator:
    """
    Simulates various impact types in AirSim using physics-based methods.
    
    Think of this as a "chaos generator" - it creates realistic disturbances
    that the PPO model must learn to recover from.
    
    Analogy: Like throwing different objects at a drone - rocks (sharp),
    fans (sustained), spinning discs (rotational), cutting power (free-fall).
    """
    
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self.active_impact = None
        self.impact_start_time = None
        self.impact_params = None
        
    def apply_impact(self, 
                     impact_type: ImpactType = None, 
                     magnitude: float = None,
                     direction: np.ndarray = None) -> ImpactParameters:
        """
        Apply a random or specified impact to the drone.
        
        Args:
            impact_type: Type of impact (random if None)
            magnitude: Impact strength 1-3 (random if None)
            direction: Impact direction (random if None)
            
        Returns:
            ImpactParameters describing the applied impact
        """
        # Random selection if not specified
        if impact_type is None:
            # Weighted probabilities based on real-world likelihood
            impact_type = np.random.choice(
                list(ImpactType),
                p=[0.4, 0.35, 0.15, 0.1]  # Sharp, Sustained, Rotational, Free-fall
            )
        
        if magnitude is None:
            # Random magnitude: 1.0 (light) to 3.0 (severe)
            magnitude = np.random.uniform(1.0, 3.0)
        
        if direction is None:
            # Random direction (normalize)
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Apply the specific impact type
        if impact_type == ImpactType.SHARP_COLLISION:
            params = self._apply_sharp_collision(magnitude, direction)
        elif impact_type == ImpactType.SUSTAINED_FORCE:
            params = self._apply_sustained_force(magnitude, direction)
        elif impact_type == ImpactType.ROTATIONAL:
            params = self._apply_rotational_disturbance(magnitude, direction)
        elif impact_type == ImpactType.FREE_FALL:
            params = self._apply_free_fall(magnitude)
        else:
            raise ValueError(f"Unknown impact type: {impact_type}")
        
        self.active_impact = impact_type
        self.impact_start_time = time.time()
        self.impact_params = params
        
        return params
    
    def _apply_sharp_collision(self, magnitude: float, direction: np.ndarray) -> ImpactParameters:
        """
        Simulate sharp collision (bird strike, wall hit).
        
        Characteristics:
        - High jerk (>30 m/s³)
        - Brief duration (<0.2s)
        - Directional impulse
        
        Physics: Instantaneous momentum transfer
        """
        # Get current state
        state = self.client.getMultirotorState()
        current_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        
        # Calculate collision impulse
        # Magnitude determines impact force (m/s change)
        impulse_strength = magnitude * 5.0  # 5-15 m/s velocity change
        impulse = direction * impulse_strength
        
        # Apply impulse by setting velocity directly
        new_vel = current_vel + impulse
        
        # Use moveByVelocityAsync for brief impulse
        duration = 0.1 + np.random.uniform(0, 0.1)  # 0.1-0.2s
        
        self.client.moveByVelocityAsync(
            vx=float(new_vel[0]),
            vy=float(new_vel[1]),
            vz=float(new_vel[2]),
            duration=duration
        )
        
        # Small random rotation from asymmetric hit
        angular_component = np.random.uniform(-1.0, 1.0) * magnitude
        
        return ImpactParameters(
            impact_type=ImpactType.SHARP_COLLISION,
            magnitude=magnitude,
            direction=direction,
            duration=duration,
            angular_component=angular_component
        )
    
    def _apply_sustained_force(self, magnitude: float, direction: np.ndarray) -> ImpactParameters:
        """
        Simulate sustained force (wind gust, continuous push).
        
        Characteristics:
        - Low jerk (<10 m/s³)
        - Long duration (1-3s)
        - Continuous acceleration
        
        Physics: Constant external force (like wind)
        """
        # Wind simulation
        wind_strength = magnitude * 5.0  # 5-15 m/s wind
        wind_vector = direction * wind_strength
        
        # Apply wind using AirSim's wind API
        self.client.simSetWind(airsim.Vector3r(
            wind_vector[0],
            wind_vector[1],
            wind_vector[2]
        ))
        
        # Duration varies
        duration = 1.0 + np.random.uniform(0, 2.0)  # 1-3 seconds
        
        return ImpactParameters(
            impact_type=ImpactType.SUSTAINED_FORCE,
            magnitude=magnitude,
            direction=direction,
            duration=duration,
            angular_component=0.0
        )
    
    def _apply_rotational_disturbance(self, magnitude: float, direction: np.ndarray) -> ImpactParameters:
        """
        Simulate rotational disturbance (asymmetric collision, prop failure).
        
        Characteristics:
        - High angular acceleration (>100°/s²)
        - Torque imbalance
        - Spinning motion
        
        Physics: Asymmetric force causing rotation
        """
        # Get current state
        state = self.client.getMultirotorState()
        
        # Apply both linear impulse AND rotational component
        # Linear part (smaller than collision)
        impulse_strength = magnitude * 2.0
        impulse = direction * impulse_strength
        
        current_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ])
        
        new_vel = current_vel + impulse
        
        # Rotational part - significant angular velocity
        angular_strength = magnitude * 3.0  # rad/s
        angular_component = angular_strength
        
        # Apply combined effect
        duration = 0.3 + np.random.uniform(0, 0.3)  # 0.3-0.6s
        
        # Apply linear velocity
        self.client.moveByVelocityAsync(
            vx=float(new_vel[0]),
            vy=float(new_vel[1]),
            vz=float(new_vel[2]),
            duration=duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(np.degrees(angular_component)))
        )
        
        # Also add some wind for realism
        wind = direction * magnitude * 3.0
        self.client.simSetWind(airsim.Vector3r(wind[0], wind[1], wind[2]))
        
        return ImpactParameters(
            impact_type=ImpactType.ROTATIONAL,
            magnitude=magnitude,
            direction=direction,
            duration=duration,
            angular_component=angular_component
        )
    
    def _apply_free_fall(self, magnitude: float) -> ImpactParameters:
        """
        Simulate free-fall (complete thrust loss, power failure).
        
        Characteristics:
        - Sudden vertical acceleration drop
        - Loss of control authority
        - Downward trajectory
        
        Physics: Gravity takes over
        """
        # Simulate by applying strong downward wind
        # This mimics loss of thrust
        fall_strength = magnitude * 8.0  # Strong downward force
        
        self.client.simSetWind(airsim.Vector3r(0, 0, fall_strength))
        
        duration = 0.5 + np.random.uniform(0, 0.5)  # 0.5-1.0s
        
        return ImpactParameters(
            impact_type=ImpactType.FREE_FALL,
            magnitude=magnitude,
            direction=np.array([0, 0, 1]),  # Downward
            duration=duration,
            angular_component=0.0
        )
    
    def clear_impact(self):
        """
        Clear all active impacts (reset wind, etc.)
        """
        # Reset wind
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        
        self.active_impact = None
        self.impact_start_time = None
        self.impact_params = None
    
    def get_time_since_impact(self) -> float:
        """
        Get time elapsed since impact started.
        """
        if self.impact_start_time is None:
            return 0.0
        return time.time() - self.impact_start_time
    
    def should_clear_impact(self) -> bool:
        """
        Check if impact duration has expired.
        """
        if self.impact_params is None:
            return False
        
        return self.get_time_since_impact() >= self.impact_params.duration
    
    def update(self):
        """
        Update impact simulation (call each step).
        Automatically clears impact when duration expires.
        """
        if self.should_clear_impact():
            self.clear_impact()


# Test the simulator
if __name__ == "__main__":
    print("="*70)
    print("IMPACT SIMULATOR TEST")
    print("="*70)
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff
    print("\n🚁 Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)
    
    # Hover at 5m
    client.moveToZAsync(-5, 2).join()
    print("✓ Hovering at 5m altitude")
    time.sleep(1)
    
    # Create simulator
    simulator = ImpactSimulator(client)
    
    # Test each impact type
    impact_types = [
        (ImpactType.SHARP_COLLISION, "💥 Sharp Collision (Bird/Wall)"),
        (ImpactType.SUSTAINED_FORCE, "🌬️  Sustained Force (Wind)"),
        (ImpactType.ROTATIONAL, "🌀 Rotational Disturbance (Asymmetric)"),
        (ImpactType.FREE_FALL, "⚡ Free Fall (Thrust Loss)")
    ]
    
    for impact_type, description in impact_types:
        print(f"\n{'-'*70}")
        print(f"Testing: {description}")
        print(f"{'-'*70}")
        
        # Apply impact
        params = simulator.apply_impact(
            impact_type=impact_type,
            magnitude=2.0,
            direction=np.array([1, 0, 0])
        )
        
        print(f"Applied: {params.impact_type.value}")
        print(f"Magnitude: {params.magnitude:.2f}")
        print(f"Direction: {params.direction}")
        print(f"Duration: {params.duration:.2f}s")
        print(f"Angular: {params.angular_component:.2f} rad/s")
        
        # Wait for impact duration + recovery time
        time.sleep(params.duration + 2.0)
        
        # Update simulator (clears impact)
        simulator.update()
        print("✓ Impact cleared, drone recovering...")
        
        # Return to hover position
        client.moveToZAsync(-5, 2).join()
        time.sleep(2)
        print("✓ Returned to hover")
    
    # Land
    print("\n🛬 Landing...")
    client.landAsync().join()
    
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n" + "="*70)
    print("✅ Impact simulation test complete!")
    print("="*70)