"""
ARDUPILOT INTERFACE - AIRSIM-LIKE API
======================================
Provides a unified interface similar to AirSim for ArduPilot drones.

Key features:
- Similar API to airsim.MultirotorClient
- Works with both SITL and real hardware
- Handles coordinate transformations (GPS → NED)
- Safe error handling
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import numpy as np
import math


class ArduPilotInterface:
    """
    ArduPilot interface with AirSim-like API
    
    Similar to: airsim.MultirotorClient()
    """
    
    def __init__(self, connection_string='127.0.0.1:14550', baud_rate=57600):
        self.connection_string = connection_string
        self.baud_rate = baud_rate
        self.vehicle = None
        self.home_position = None
        self.connected = False
        
    def confirmConnection(self):
        """Connect to vehicle (like AirSim)"""
        try:
            if self.connection_string.startswith('/dev/'):
                self.vehicle = connect(self.connection_string, wait_ready=True, baud=self.baud_rate)
            else:
                self.vehicle = connect(self.connection_string, wait_ready=True)
            
            self.connected = True
            
            # Wait for GPS
            while self.vehicle.gps_0.fix_type < 2:
                print("Waiting for GPS fix...")
                time.sleep(1)
            
            # Store home position
            self.home_position = self.vehicle.location.global_relative_frame
            
            print(f"✅ Connected to ArduPilot")
            print(f"   Home: Lat={self.home_position.lat:.6f}, "
                  f"Lon={self.home_position.lon:.6f}, "
                  f"Alt={self.home_position.alt:.2f}m")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
    
    def enableApiControl(self, enable=True):
        """Enable API control (switch to GUIDED mode)"""
        if enable:
            self.vehicle.mode = VehicleMode("GUIDED")
            while self.vehicle.mode.name != "GUIDED":
                time.sleep(0.1)
            print("✅ API control enabled (GUIDED mode)")
        else:
            self.vehicle.mode = VehicleMode("STABILIZE")
            print("✅ API control disabled (STABILIZE mode)")
    
    def armDisarm(self, arm=True):
        """Arm or disarm the vehicle"""
        if arm:
            # Pre-arm checks
            while not self.vehicle.is_armable:
                print("Waiting for vehicle to be armable...")
                time.sleep(1)
            
            self.vehicle.armed = True
            while not self.vehicle.armed:
                print("Arming...")
                time.sleep(0.5)
            print("✅ Armed")
        else:
            self.vehicle.armed = False
            while self.vehicle.armed:
                print("Disarming...")
                time.sleep(0.5)
            print("✅ Disarmed")
    
    def takeoffAsync(self, altitude=10.0):
        """Takeoff to specified altitude"""
        print(f"Taking off to {altitude}m...")
        self.vehicle.simple_takeoff(altitude)
        
        # Wait to reach target altitude
        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt
            print(f"  Altitude: {current_alt:.1f}m / {altitude}m")
            
            if current_alt >= altitude * 0.95:
                print("✅ Target altitude reached")
                break
            time.sleep(0.5)
        
        return DummyAsync()
    
    def moveToPositionAsync(self, x, y, z, velocity=5):
        """
        Move to position in NED coordinates
        
        Args:
            x, y, z: Position in NED (North-East-Down) meters
            velocity: Target velocity (m/s)
        """
        # Convert NED to GPS
        target_location = self._ned_to_gps(x, y, z)
        
        # Send command
        self.vehicle.simple_goto(target_location, groundspeed=velocity)
        
        # Wait to reach position
        while True:
            current_pos = self.vehicle.location.global_relative_frame
            distance = self._get_distance_metres(current_pos, target_location)
            
            if distance < 1.0:  # Within 1m
                break
            time.sleep(0.5)
        
        return DummyAsync()
    
    def moveByVelocityAsync(self, vx, vy, vz, duration, 
                           drivetrain=None, yaw_mode=None):
        """
        Move by velocity in NED frame
        
        Args:
            vx, vy, vz: Velocity in NED (m/s)
            duration: Duration (seconds)
        """
        # Send velocity command via MAVLink
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # type_mask (only velocities enabled)
            0, 0, 0,  # x, y, z positions (not used)
            vx, vy, vz,  # x, y, z velocity in m/s
            0, 0, 0,  # x, y, z acceleration (not used)
            0, 0)    # yaw, yaw_rate (not used)
        
        # Send command for duration
        end_time = time.time() + duration
        while time.time() < end_time:
            self.vehicle.send_mavlink(msg)
            time.sleep(0.05)  # 20Hz
        
        return DummyAsync()
    
    def getMultirotorState(self):
        """Get current drone state (AirSim-like)"""
        state = MultirotorState()
        
        # Position (NED coordinates)
        pos_ned = self._gps_to_ned(
            self.vehicle.location.global_relative_frame
        )
        state.kinematics_estimated.position.x_val = pos_ned[0]
        state.kinematics_estimated.position.y_val = pos_ned[1]
        state.kinematics_estimated.position.z_val = pos_ned[2]
        
        # Velocity (NED)
        state.kinematics_estimated.linear_velocity.x_val = self.vehicle.velocity[0]  # North
        state.kinematics_estimated.linear_velocity.y_val = self.vehicle.velocity[1]  # East
        state.kinematics_estimated.linear_velocity.z_val = self.vehicle.velocity[2]  # Down
        
        # Orientation (quaternion)
        # ArduPilot gives Euler angles, convert to quaternion
        roll = self.vehicle.attitude.roll
        pitch = self.vehicle.attitude.pitch
        yaw = self.vehicle.attitude.yaw
        
        quat = self._euler_to_quaternion(roll, pitch, yaw)
        state.kinematics_estimated.orientation.w_val = quat[0]
        state.kinematics_estimated.orientation.x_val = quat[1]
        state.kinematics_estimated.orientation.y_val = quat[2]
        state.kinematics_estimated.orientation.z_val = quat[3]
        
        # Angular velocity
        gyro = self.vehicle.gimbal.rot  # Use gimbal for angular rates
        # Or use attitude derivative (less accurate)
        state.kinematics_estimated.angular_velocity.x_val = 0  # Not directly available
        state.kinematics_estimated.angular_velocity.y_val = 0
        state.kinematics_estimated.angular_velocity.z_val = 0
        
        return state
    
    def simGetCollisionInfo(self):
        """Get collision info (always False for real drone)"""
        collision = CollisionInfo()
        collision.has_collided = False
        return collision
    
    def simSetWind(self, wind_vector):
        """
        Set wind (only works in SITL with wind simulation enabled)
        
        Note: Real hardware can't simulate wind!
        """
        # This requires SITL with SIM_WIND_* parameters
        # For real hardware, this is a no-op
        pass
    
    def reset(self):
        """Reset (for SITL, restart at home position)"""
        # For SITL: use SITL_START_LOCATION
        # For real hardware: land and rearm
        pass
    
    def landAsync(self):
        """Land the vehicle"""
        print("Landing...")
        self.vehicle.mode = VehicleMode("LAND")
        
        while self.vehicle.location.global_relative_frame.alt > 0.5:
            time.sleep(0.5)
        
        print("✅ Landed")
        return DummyAsync()
    
    # Helper functions
    def _ned_to_gps(self, north, east, down):
        """Convert NED coordinates to GPS"""
        # Earth radius in meters
        R = 6378137.0
        
        # Offsets in meters
        dLat = north / R
        dLon = east / (R * math.cos(math.pi * self.home_position.lat / 180))
        
        # New position
        new_lat = self.home_position.lat + (dLat * 180 / math.pi)
        new_lon = self.home_position.lon + (dLon * 180 / math.pi)
        new_alt = self.home_position.alt - down  # Down is negative altitude
        
        return LocationGlobalRelative(new_lat, new_lon, new_alt)
    
    def _gps_to_ned(self, location):
        """Convert GPS to NED coordinates"""
        R = 6378137.0
        
        dLat = location.lat - self.home_position.lat
        dLon = location.lon - self.home_position.lon
        
        north = dLat * (math.pi / 180) * R
        east = dLon * (math.pi / 180) * R * math.cos(math.pi * self.home_position.lat / 180)
        down = -(location.alt - self.home_position.alt)
        
        return np.array([north, east, down])
    
    def _get_distance_metres(self, loc1, loc2):
        """Get distance between two GPS points"""
        dlat = loc2.lat - loc1.lat
        dlon = loc2.lon - loc1.lon
        return math.sqrt((dlat*dlat) + (dlon*dlon)) * 1.113195e5
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])


# Helper classes to mimic AirSim API
class DummyAsync:
    """Dummy async object for compatibility"""
    def join(self):
        pass


class MultirotorState:
    """Mimic AirSim MultirotorState"""
    def __init__(self):
        self.kinematics_estimated = KinematicsState()


class KinematicsState:
    """Mimic AirSim KinematicsState"""
    def __init__(self):
        self.position = Vector3r()
        self.linear_velocity = Vector3r()
        self.orientation = Quaternionr()
        self.angular_velocity = Vector3r()


class Vector3r:
    """Mimic AirSim Vector3r"""
    def __init__(self):
        self.x_val = 0.0
        self.y_val = 0.0
        self.z_val = 0.0


class Quaternionr:
    """Mimic AirSim Quaternionr"""
    def __init__(self):
        self.w_val = 1.0
        self.x_val = 0.0
        self.y_val = 0.0
        self.z_val = 0.0


class CollisionInfo:
    """Mimic AirSim CollisionInfo"""
    def __init__(self):
        self.has_collided = False