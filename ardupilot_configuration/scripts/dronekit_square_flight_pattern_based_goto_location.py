"""
# Make sure ArduPilot SITL is running first
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
--sim-address=172.23.128.1 \
--out=127.0.0.1:14550

# In another WSL2 terminal, run the script
cd ~/your_project
python3 square_flight.py

"""

#!/usr/bin/env python3
"""
Square Flight Pattern with DroneKit
====================================

This script demonstrates basic drone control:
1. Takeoff to 10m altitude
2. Move forward 5m
3. Move left 5m
4. Move backward 5m
5. Move right 5m (back to start)
6. Land

Author: Created for PhD research on drone behavioral cloning
Date: November 16, 2025
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

CONNECTION_STRING = 'udp:127.0.0.1:14550'
TAKEOFF_ALTITUDE = 10  # meters
MOVEMENT_DISTANCE = 5  # meters
MOVEMENT_SPEED = 2     # m/s
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


def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobalRelative object containing the latitude/longitude 
    `dNorth` and `dEast` metres from the specified `original_location`.
    
    Think of this like giving directions: "From here, go 5m north and 3m east"
    
    The function uses an approximation valid for small distances.
    """
    earth_radius = 6378137.0  # Radius of Earth in metres
    
    # Coordinate offsets in radians
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))
    
    # New position in decimal degrees
    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    
    return LocationGlobalRelative(newlat, newlon, original_location.alt)


def get_distance_metres(location1, location2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.
    
    This uses the Haversine formula - think of it like using Google Maps 
    to calculate "as the crow flies" distance.
    """
    dlat = location2.lat - location1.lat
    dlon = location2.lon - location1.lon
    
    return math.sqrt((dlat * dlat) + (dlon * dlon)) * 1.113195e5


def goto_position_target_local_ned(vehicle, north, east, down):
    """
    Send command to move the vehicle to a specified position (in metres) 
    relative to the current position.
    
    Think of this like telling the drone: "From where you are now, 
    move 5m forward (north), 0m sideways (east), 0m up/down (down)"
    
    The velocity parameter allows us to control the speed.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        north, east, down,   # x, y, z positions
        0, 0, 0,  # x, y, z velocity (not used)
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)     # yaw, yaw_rate (not used)
    
    vehicle.send_mavlink(msg)


def goto_location(vehicle, target_location, speed):
    """
    Move to a target location at specified speed.
    
    Think of this like using GPS navigation: "Drive to this coordinate at 50 km/h"
    """
    print(f"\nMoving to target location...")
    print(f"  Current: Lat={vehicle.location.global_relative_frame.lat:.6f}, "
          f"Lon={vehicle.location.global_relative_frame.lon:.6f}")
    print(f"  Target:  Lat={target_location.lat:.6f}, "
          f"Lon={target_location.lon:.6f}")
    
    # Set groundspeed
    vehicle.groundspeed = speed
    
    # Go to target
    vehicle.simple_goto(target_location)
    
    # Wait until vehicle reaches target
    while True:
        current_location = vehicle.location.global_relative_frame
        distance = get_distance_metres(current_location, target_location)
        
        print(f"  Distance to target: {distance:.2f}m", end='\r')
        
        # Consider reached if within 1m
        if distance < 1.0:
            print(f"\n✓ Reached target location!                    ")
            break
        
        time.sleep(0.5)


def fly_square_pattern(vehicle, side_length, speed):
    """
    Fly in a square pattern:
    1. Forward (North)
    2. Left (West) 
    3. Backward (South)
    4. Right (East)
    
    Think of it like drawing a square on the ground from above.
    """
    print("\n" + "=" * 60)
    print("FLYING SQUARE PATTERN")
    print("=" * 60)
    print(f"Side length: {side_length}m")
    print(f"Speed: {speed}m/s")
    print("=" * 60)
    
    # Get starting position
    start_location = vehicle.location.global_relative_frame
    print(f"\nStarting position recorded:")
    print(f"  Lat: {start_location.lat:.6f}")
    print(f"  Lon: {start_location.lon:.6f}")
    print(f"  Alt: {start_location.alt:.1f}m")
    
    # === MOVE 1: FORWARD (NORTH) ===
    print("\n" + "-" * 60)
    print("MOVE 1: FORWARD (NORTH) - 5 metres")
    print("-" * 60)
    target1 = get_location_metres(start_location, side_length, 0)
    goto_location(vehicle, target1, speed)
    print(f"Waiting {WAIT_TIME}s at corner 1...")
    time.sleep(WAIT_TIME)
    
    # === MOVE 2: LEFT (WEST) ===
    print("\n" + "-" * 60)
    print("MOVE 2: LEFT (WEST) - 5 metres")
    print("-" * 60)
    target2 = get_location_metres(target1, 0, -side_length)
    goto_location(vehicle, target2, speed)
    print(f"Waiting {WAIT_TIME}s at corner 2...")
    time.sleep(WAIT_TIME)
    
    # === MOVE 3: BACKWARD (SOUTH) ===
    print("\n" + "-" * 60)
    print("MOVE 3: BACKWARD (SOUTH) - 5 metres")
    print("-" * 60)
    target3 = get_location_metres(target2, -side_length, 0)
    goto_location(vehicle, target3, speed)
    print(f"Waiting {WAIT_TIME}s at corner 3...")
    time.sleep(WAIT_TIME)
    
    # === MOVE 4: RIGHT (EAST) - BACK TO START ===
    print("\n" + "-" * 60)
    print("MOVE 4: RIGHT (EAST) - 5 metres (back to start)")
    print("-" * 60)
    target4 = get_location_metres(target3, 0, side_length)
    goto_location(vehicle, target4, speed)
    print(f"Waiting {WAIT_TIME}s at starting position...")
    time.sleep(WAIT_TIME)
    
    # Verify we're back at start
    final_location = vehicle.location.global_relative_frame
    distance_from_start = get_distance_metres(start_location, final_location)
    print(f"\n✓ Square pattern complete!")
    print(f"  Distance from starting point: {distance_from_start:.2f}m")
    
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
    print("█" + "  DRONEKIT SQUARE FLIGHT PATTERN".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print("\nThis script will:")
    print("  1. Connect to ArduPilot SITL")
    print("  2. Arm and takeoff to 10m")
    print("  3. Fly a 5m x 5m square pattern")
    print("  4. Return to start and land")
    print("\nMake sure ArduPilot SITL and AirSim are running!")
    print("Press Ctrl+C at any time to abort.\n")
    
    input("Press ENTER to start...")
    
    vehicle = None
    
    try:
        # Connect to vehicle
        vehicle = connect_vehicle()
        
        # Takeoff
        arm_and_takeoff(vehicle, TAKEOFF_ALTITUDE)
        
        # Fly square pattern
        fly_square_pattern(vehicle, MOVEMENT_DISTANCE, MOVEMENT_SPEED)
        
        # Land
        land_vehicle(vehicle)
        
        print("\n" + "=" * 60)
        print("MISSION COMPLETE!")
        print("=" * 60)
        print("The drone successfully:")
        print("  ✓ Took off to 10m")
        print("  ✓ Flew forward 5m")
        print("  ✓ Flew left 5m")
        print("  ✓ Flew backward 5m")
        print("  ✓ Flew right 5m")
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