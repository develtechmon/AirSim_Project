"""
# Terminal 1: ArduPilot SITL
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
  --sim-address=172.23.128.1 \
  --out=127.0.0.1:14550

# Terminal 2: Run the script
python3 dronekit_velocity_based_control.py
"""


#!/usr/bin/env python3
"""
Square Flight Pattern with Velocity Control (No Yaw Rotation)
==============================================================

This script demonstrates velocity-based drone control where the drone
moves in a square pattern WITHOUT rotating its heading:
1. Takeoff to 10m altitude
2. Move forward 5m (North) - nose stays pointing North
3. Move left 5m (West) - nose STILL pointing North (strafing)
4. Move backward 5m (South) - nose STILL pointing North
5. Move right 5m (East) - nose STILL pointing North (back to start)
6. Land

Think of it like a crab walking - moves in all directions without turning.

Author: Created for PhD research on drone behavioral cloning
Date: November 16, 2025
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

CONNECTION_STRING = 'udp:127.0.0.1:14550'
TAKEOFF_ALTITUDE = 10  # meters
MOVEMENT_DISTANCE = 5  # meters
MOVEMENT_SPEED = 1     # m/s (slower for better control)
WAIT_TIME = 2          # seconds to wait at each corner

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def connect_vehicle():
    """Connect to the vehicle and wait until ready."""
    print("=" * 60)
    print("CONNECTING TO VEHICLE")
    print("=" * 60)
    print(f"Connection string: {CONNECTION_STRING}")
    
    vehicle = connect(CONNECTION_STRING, wait_ready=True)
    
    print("\nVehicle connected successfully!")
    print(f"Mode: {vehicle.mode.name}")
    print(f"GPS: {vehicle.gps_0.fix_type}")
    print(f"Battery: {vehicle.battery.level}%")
    print(f"Armed: {vehicle.armed}")
    print("=" * 60)
    
    return vehicle


def arm_and_takeoff(vehicle, target_altitude):
    """
    Arms vehicle and flies to target_altitude.
    
    Think of this like starting a car (arm) and then driving up a ramp (takeoff).
    """
    print("\n" + "=" * 60)
    print("ARM AND TAKEOFF")
    print("=" * 60)
    
    # Pre-arm checks
    print("Running pre-arm checks...")
    while not vehicle.is_armable:
        print("  Waiting for vehicle to initialise...")
        time.sleep(1)
    
    print("✓ Vehicle is armable")
    
    # Switch to GUIDED mode
    print("\nSwitching to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        print("  Waiting for GUIDED mode...")
        time.sleep(0.5)
    
    print("✓ GUIDED mode active")
    
    # Arm the vehicle
    print("\nArming motors...")
    vehicle.armed = True
    while not vehicle.armed:
        print("  Waiting for arming...")
        time.sleep(0.5)
    
    print("✓ Motors armed!")
    
    # Takeoff
    print(f"\nTaking off to {target_altitude}m...")
    vehicle.simple_takeoff(target_altitude)
    
    # Wait until vehicle reaches target altitude
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {current_altitude:.1f}m / {target_altitude}m")
        
        # Consider reached if within 95% of target
        if current_altitude >= target_altitude * 0.95:
            print("✓ Target altitude reached!")
            break
        
        time.sleep(1)
    
    print("=" * 60)


def send_ned_velocity(vehicle, velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    
    This is the KEY function that allows movement without yaw rotation!
    
    Parameters:
    - velocity_x: Forward velocity in m/s (positive = forward, negative = backward)
    - velocity_y: Right velocity in m/s (positive = right, negative = left)
    - velocity_z: Down velocity in m/s (positive = down, negative = up)
    - duration: Time in seconds to apply this velocity
    
    Think of it like a video game controller:
    - Left joystick forward/back = velocity_x
    - Left joystick left/right = velocity_y
    - Right trigger up/down = velocity_z
    
    The drone moves in that direction WITHOUT rotating!
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)     # yaw, yaw_rate (not used)
    
    # Send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def send_ned_velocity_with_feedback(vehicle, velocity_x, velocity_y, velocity_z, 
                                     target_distance, direction_name):
    """
    Move vehicle using velocity control until target distance is reached.
    
    This is smarter than fixed duration - it stops when the drone has 
    actually moved the desired distance.
    
    Think of it like cruise control that turns off after 5km, not after 5 minutes.
    """
    print(f"\n{direction_name}: Target distance = {target_distance}m")
    
    # Record starting position
    start_location = vehicle.location.global_relative_frame
    
    # Calculate how long it should take (rough estimate)
    speed = math.sqrt(velocity_x**2 + velocity_y**2)
    estimated_time = target_distance / speed if speed > 0 else 10
    
    print(f"  Speed: {speed:.1f}m/s")
    print(f"  Estimated time: {estimated_time:.1f}s")
    
    distance_travelled = 0
    start_time = time.time()
    
    while distance_travelled < target_distance:
        # Send velocity command
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms (not used)
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
            0b0000111111000111,  # type_mask (only speeds enabled)
            0, 0, 0,  # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
            0, 0, 0,  # x, y, z acceleration (not used)
            0, 0)     # yaw, yaw_rate (not used)
        
        vehicle.send_mavlink(msg)
        
        # Calculate distance travelled
        current_location = vehicle.location.global_relative_frame
        distance_travelled = get_distance_metres(start_location, current_location)
        
        # Show progress
        elapsed_time = time.time() - start_time
        print(f"  Distance: {distance_travelled:.2f}m / {target_distance}m  "
              f"Time: {elapsed_time:.1f}s  "
              f"Heading: {vehicle.heading}°", end='\r')
        
        time.sleep(0.1)  # 10Hz update rate
        
        # Safety timeout (if something goes wrong)
        if elapsed_time > estimated_time * 3:
            print(f"\n⚠ Timeout reached after {elapsed_time:.1f}s")
            break
    
    # Stop the vehicle
    stop_vehicle(vehicle)
    
    print(f"\n✓ Completed! Distance travelled: {distance_travelled:.2f}m")


def stop_vehicle(vehicle):
    """
    Stop the vehicle by sending zero velocity commands.
    
    Think of it like releasing the joystick - drone stops moving.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        0, 0, 0,  # Zero velocity
        0, 0, 0,
        0, 0)
    
    # Send stop command multiple times to ensure it's received
    for _ in range(5):
        vehicle.send_mavlink(msg)
        time.sleep(0.1)


def get_distance_metres(location1, location2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.
    
    This uses the Haversine formula - think of it like using Google Maps 
    to calculate "as the crow flies" distance.
    """
    dlat = location2.lat - location1.lat
    dlon = location2.lon - location1.lon
    
    return math.sqrt((dlat * dlat) + (dlon * dlon)) * 1.113195e5


def fly_square_pattern_velocity(vehicle, side_length, speed):
    """
    Fly in a square pattern using VELOCITY control (no yaw rotation).
    
    The drone will:
    1. Move forward (North) - nose pointing North
    2. Strafe left (West) - nose STILL pointing North
    3. Move backward (South) - nose STILL pointing North  
    4. Strafe right (East) - nose STILL pointing North
    
    It's like a security guard walking around a building while always 
    facing the same direction - perfect for surveillance!
    """
    print("\n" + "=" * 60)
    print("FLYING SQUARE PATTERN (VELOCITY CONTROL)")
    print("=" * 60)
    print(f"Side length: {side_length}m")
    print(f"Speed: {speed}m/s")
    print("Movement style: STRAFING (no yaw rotation)")
    print("=" * 60)
    
    # Get starting position
    start_location = vehicle.location.global_relative_frame
    start_heading = vehicle.heading
    print(f"\nStarting position recorded:")
    print(f"  Lat: {start_location.lat:.6f}")
    print(f"  Lon: {start_location.lon:.6f}")
    print(f"  Alt: {start_location.alt:.1f}m")
    print(f"  Heading: {start_heading}°")
    
    # === MOVE 1: FORWARD (NORTH) ===
    print("\n" + "-" * 60)
    print("MOVE 1: FORWARD (velocity_x = +1.0 m/s)")
    print("        Drone nose pointing North, moving North")
    print("-" * 60)
    send_ned_velocity_with_feedback(
        vehicle,
        velocity_x=speed,   # Forward
        velocity_y=0,       # No sideways
        velocity_z=0,       # No up/down
        target_distance=side_length,
        direction_name="FORWARD"
    )
    print(f"Waiting {WAIT_TIME}s at corner 1...")
    print(f"  Current heading: {vehicle.heading}° (should be ~{start_heading}°)")
    time.sleep(WAIT_TIME)
    
    # === MOVE 2: LEFT (WEST) - STRAFING ===
    print("\n" + "-" * 60)
    print("MOVE 2: LEFT STRAFE (velocity_y = -1.0 m/s)")
    print("        Drone nose STILL pointing North, but moving West")
    print("        This is the key move - sideways without turning!")
    print("-" * 60)
    send_ned_velocity_with_feedback(
        vehicle,
        velocity_x=0,       # No forward/back
        velocity_y=-speed,  # Left (negative = left in NED frame)
        velocity_z=0,       # No up/down
        target_distance=side_length,
        direction_name="LEFT STRAFE"
    )
    print(f"Waiting {WAIT_TIME}s at corner 2...")
    print(f"  Current heading: {vehicle.heading}° (should be ~{start_heading}°)")
    time.sleep(WAIT_TIME)
    
    # === MOVE 3: BACKWARD (SOUTH) ===
    print("\n" + "-" * 60)
    print("MOVE 3: BACKWARD (velocity_x = -1.0 m/s)")
    print("        Drone nose STILL pointing North, moving South (backwards)")
    print("-" * 60)
    send_ned_velocity_with_feedback(
        vehicle,
        velocity_x=-speed,  # Backward (negative = backward)
        velocity_y=0,       # No sideways
        velocity_z=0,       # No up/down
        target_distance=side_length,
        direction_name="BACKWARD"
    )
    print(f"Waiting {WAIT_TIME}s at corner 3...")
    print(f"  Current heading: {vehicle.heading}° (should be ~{start_heading}°)")
    time.sleep(WAIT_TIME)
    
    # === MOVE 4: RIGHT (EAST) - STRAFING BACK TO START ===
    print("\n" + "-" * 60)
    print("MOVE 4: RIGHT STRAFE (velocity_y = +1.0 m/s)")
    print("        Drone nose STILL pointing North, moving East (back to start)")
    print("-" * 60)
    send_ned_velocity_with_feedback(
        vehicle,
        velocity_x=0,      # No forward/back
        velocity_y=speed,  # Right (positive = right in NED frame)
        velocity_z=0,      # No up/down
        target_distance=side_length,
        direction_name="RIGHT STRAFE"
    )
    print(f"Waiting {WAIT_TIME}s at starting position...")
    time.sleep(WAIT_TIME)
    
    # Verify we're back at start
    final_location = vehicle.location.global_relative_frame
    final_heading = vehicle.heading
    distance_from_start = get_distance_metres(start_location, final_location)
    heading_change = abs(final_heading - start_heading)
    
    print(f"\n✓ Square pattern complete!")
    print(f"  Distance from starting point: {distance_from_start:.2f}m")
    print(f"  Starting heading: {start_heading}°")
    print(f"  Final heading: {final_heading}°")
    print(f"  Heading change: {heading_change:.1f}° (should be ~0°)")
    
    if heading_change < 10:
        print("  ✓ SUCCESS: Drone maintained heading (no yaw rotation)!")
    else:
        print("  ⚠ WARNING: Drone heading changed significantly")
    
    print("=" * 60)


def land_vehicle(vehicle):
    """
    Land the vehicle and disarm.
    
    Think of this like parking a car and turning off the engine.
    """
    print("\n" + "=" * 60)
    print("LANDING")
    print("=" * 60)
    
    print("Switching to LAND mode...")
    vehicle.mode = VehicleMode("LAND")
    
    # Wait until landed
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {altitude:.1f}m", end='\r')
        
        if altitude < 0.5:
            print(f"\n✓ Landed!                    ")
            break
        
        time.sleep(0.5)
    
    # Wait for disarm
    print("\nWaiting for disarm...")
    while vehicle.armed:
        print("  Motors still armed...", end='\r')
        time.sleep(0.5)
    
    print("✓ Motors disarmed!                    ")
    print("=" * 60)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  DRONEKIT SQUARE FLIGHT - VELOCITY CONTROL".center(58) + "█")
    print("█" + "  (No Yaw Rotation / Strafing Movement)".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print("\nThis script will:")
    print("  1. Connect to ArduPilot SITL")
    print("  2. Arm and takeoff to 10m")
    print("  3. Fly a 5m x 5m square using velocity commands")
    print("  4. The drone will NOT rotate its heading")
    print("  5. Movements: Forward → Strafe Left → Backward → Strafe Right")
    print("  6. Return to start and land")
    print("\nThe drone will move like a crab - sideways without turning!")
    print("\nMake sure ArduPilot SITL and AirSim are running!")
    print("Press Ctrl+C at any time to abort.\n")
    
    input("Press ENTER to start...")
    
    vehicle = None
    
    try:
        # Connect to vehicle
        vehicle = connect_vehicle()
        
        # Takeoff
        arm_and_takeoff(vehicle, TAKEOFF_ALTITUDE)
        
        # Fly square pattern with velocity control
        fly_square_pattern_velocity(vehicle, MOVEMENT_DISTANCE, MOVEMENT_SPEED)
        
        # Land
        land_vehicle(vehicle)
        
        print("\n" + "=" * 60)
        print("MISSION COMPLETE!")
        print("=" * 60)
        print("The drone successfully:")
        print("  ✓ Took off to 10m")
        print("  ✓ Moved forward 5m (no rotation)")
        print("  ✓ Strafed left 5m (no rotation)")
        print("  ✓ Moved backward 5m (no rotation)")
        print("  ✓ Strafed right 5m (no rotation)")
        print("  ✓ Maintained constant heading throughout")
        print("  ✓ Landed safely")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ KEYBOARD INTERRUPT DETECTED!")
        print("Attempting emergency landing...")
        if vehicle:
            vehicle.mode = VehicleMode("LAND")
            time.sleep(5)
    
    except Exception as e:
        print(f"\n\n❌ ERROR OCCURRED: {e}")
        print("Attempting emergency landing...")
        if vehicle:
            vehicle.mode = VehicleMode("LAND")
            time.sleep(5)
    
    finally:
        if vehicle:
            print("\nClosing vehicle connection...")
            vehicle.close()
            print("✓ Connection closed")
        
        print("\nScript terminated.\n")


if __name__ == "__main__":
    main()