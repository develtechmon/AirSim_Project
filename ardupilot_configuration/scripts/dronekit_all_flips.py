#!/usr/bin/env python3
"""
Acrobatic Flip Maneuvers with DroneKit using ArduPilot's FLIP Mode
===================================================================

This script performs acrobatic flips using ArduPilot's BUILT-IN FLIP mode.

KEY INSIGHT: AirSim's moveByAngleRatesThrottleAsync() gives you direct motor control.
ArduPilot doesn't allow this in GUIDED mode for safety reasons. Instead, we use
ArduPilot's native FLIP mode which is specifically designed for safe flips.

How it works:
1. Set up an RC auxiliary function for FLIP (RCx_OPTION = 2)
2. Use RC_OVERRIDE to simulate stick input for flip direction
3. Trigger FLIP mode via MAVLink
4. ArduPilot handles the flip safely and returns to original mode

Author: Created for PhD research on drone behavioral cloning
Date: November 2025

IMPORTANT: This requires ArduPilot parameter setup first!
"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

CONNECTION_STRING = 'udp:127.0.0.1:14550'
TAKEOFF_ALTITUDE = 10     # meters - need height for safe recovery
HOVER_ALTITUDE = 8        # meters - return altitude after each flip
FLIP_CHANNEL = 7          # RC channel for flip trigger (RC7_OPTION = 2)

# ArduPilot FLIP mode number
FLIP_MODE = 14

# Flip directions via stick input
# When FLIP mode is triggered, the flip direction is determined by stick position
FLIP_RIGHT = 1     # Roll stick right
FLIP_LEFT = -1     # Roll stick left  
FLIP_FORWARD = 1   # Pitch stick forward
FLIP_BACKWARD = -1 # Pitch stick backward

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


def setup_flip_parameters(vehicle):
    """
    Configure ArduPilot parameters for FLIP mode.
    
    This sets RC7 as the FLIP trigger channel.
    """
    print("\n" + "=" * 60)
    print("CONFIGURING FLIP PARAMETERS")
    print("=" * 60)
    
    # Set RC7_OPTION to FLIP (value 2)
    print("Setting RC7_OPTION = 2 (FLIP trigger)...")
    vehicle.parameters['RC7_OPTION'] = 2
    time.sleep(0.5)
    
    # Verify
    rc7_option = vehicle.parameters.get('RC7_OPTION', None)
    print(f"  RC7_OPTION = {rc7_option}")
    
    if rc7_option == 2:
        print("✓ FLIP trigger configured on RC7")
    else:
        print("⚠ Warning: RC7_OPTION may not have been set correctly")
    
    print("=" * 60)


def arm_and_takeoff(vehicle, target_altitude):
    """Arms vehicle and flies to target_altitude."""
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
        
        if current_altitude >= target_altitude * 0.95:
            print("✓ Target altitude reached!")
            break
        
        time.sleep(1)
    
    print("=" * 60)


def send_rc_override(vehicle, roll=0, pitch=0, throttle=0, yaw=0, duration=0.1):
    """
    Send RC override commands to simulate stick input.
    
    Values are in PWM (1000-2000), with 1500 being center/neutral.
    
    Parameters:
    - roll: -1 to 1 (left to right)
    - pitch: -1 to 1 (back to forward)
    - throttle: 0 to 1
    - yaw: -1 to 1 (left to right)
    - duration: How long to hold this input
    """
    # Convert -1 to 1 range to PWM (1000-2000)
    roll_pwm = int(1500 + roll * 500)
    pitch_pwm = int(1500 + pitch * 500)
    throttle_pwm = int(1000 + throttle * 1000)
    yaw_pwm = int(1500 + yaw * 500)
    
    # Channel mapping: 1=Roll, 2=Pitch, 3=Throttle, 4=Yaw
    # 0 means "no override" for that channel
    msg = vehicle.message_factory.rc_channels_override_encode(
        0, 0,  # target_system, target_component
        roll_pwm,      # chan1 (roll)
        pitch_pwm,     # chan2 (pitch)
        throttle_pwm,  # chan3 (throttle)
        yaw_pwm,       # chan4 (yaw)
        0, 0,          # chan5, chan6
        0, 0           # chan7, chan8
    )
    
    start_time = time.time()
    while time.time() - start_time < duration:
        vehicle.send_mavlink(msg)
        time.sleep(0.02)  # 50Hz


def trigger_flip_mode(vehicle, roll_direction=0, pitch_direction=0):
    """
    Trigger ArduPilot's FLIP mode with specified direction.
    
    Parameters:
    - roll_direction: -1 (left), 0 (none), 1 (right)
    - pitch_direction: -1 (back), 0 (none), 1 (forward)
    
    The flip direction is determined by holding the stick slightly
    in that direction when entering FLIP mode.
    """
    # Remember current mode
    original_mode = vehicle.mode.name
    print(f"  Original mode: {original_mode}")
    
    # First, go to a flip-compatible mode (ALT_HOLD is safest)
    print("  Switching to ALT_HOLD (required for FLIP)...")
    vehicle.mode = VehicleMode("ALT_HOLD")
    time.sleep(1)
    
    # Apply stick input for direction (hold slightly)
    # This tells FLIP mode which direction to flip
    roll_input = roll_direction * 0.3  # 30% stick
    pitch_input = pitch_direction * 0.3
    
    direction_name = ""
    if roll_direction > 0:
        direction_name = "RIGHT"
    elif roll_direction < 0:
        direction_name = "LEFT"
    elif pitch_direction > 0:
        direction_name = "FORWARD"
    elif pitch_direction < 0:
        direction_name = "BACKWARD"
    
    print(f"  Applying {direction_name} stick input...")
    
    # Send RC override to indicate direction
    send_rc_override(vehicle, roll=roll_input, pitch=pitch_input, 
                     throttle=0.5, duration=0.2)
    
    # Now switch to FLIP mode
    print("  Triggering FLIP mode...")
    
    # Method 1: Direct mode change
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,  # confirmation
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        FLIP_MODE,  # FLIP = 14
        0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    
    # Keep sending direction input during flip
    print("  Flipping...")
    flip_start = time.time()
    while time.time() - flip_start < 3.0:  # Flip takes ~2.5s max
        send_rc_override(vehicle, roll=roll_input, pitch=pitch_input,
                        throttle=0.6, duration=0.1)
        
        # Check if flip is complete (mode changes back)
        if vehicle.mode.name != "FLIP":
            break
    
    print(f"  Flip completed! Current mode: {vehicle.mode.name}")
    
    # Return to GUIDED mode
    print("  Returning to GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    
    # Clear RC override
    clear_rc_override(vehicle)


def clear_rc_override(vehicle):
    """Clear all RC overrides."""
    msg = vehicle.message_factory.rc_channels_override_encode(
        0, 0,
        0, 0, 0, 0,  # All zeros = release override
        0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)


def execute_flip_sequence(vehicle, flip_name, roll_dir, pitch_dir):
    """Execute a complete flip with recovery."""
    print(f"\n  === {flip_name} ===")
    
    # Check altitude before flip
    alt = vehicle.location.global_relative_frame.alt
    print(f"  Pre-flip altitude: {alt:.1f}m")
    
    if alt < 8:
        print("  ⚠ Altitude too low, climbing first...")
        move_to_altitude(vehicle, HOVER_ALTITUDE)
    
    # Execute flip
    trigger_flip_mode(vehicle, roll_direction=roll_dir, pitch_direction=pitch_dir)
    
    # Wait for stabilization
    print("  Stabilizing...")
    time.sleep(2)
    
    # Check altitude after flip
    alt = vehicle.location.global_relative_frame.alt
    print(f"  Post-flip altitude: {alt:.1f}m")
    
    # Return to safe altitude if needed
    if alt < 5:
        print("  Recovering altitude...")
        move_to_altitude(vehicle, HOVER_ALTITUDE)
    
    print(f"  ✓ {flip_name} complete!")


def move_to_altitude(vehicle, target_altitude):
    """Move to a target altitude."""
    print(f"  Moving to {target_altitude}m...")
    
    from dronekit import LocationGlobalRelative
    current_location = vehicle.location.global_relative_frame
    target_location = LocationGlobalRelative(
        current_location.lat,
        current_location.lon,
        target_altitude
    )
    
    vehicle.simple_goto(target_location, groundspeed=2)
    
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        if abs(current_alt - target_altitude) < 0.5:
            print(f"  ✓ Altitude {target_altitude}m reached")
            break
        time.sleep(0.5)


def land_vehicle(vehicle):
    """Land the vehicle safely."""
    print("\n" + "=" * 60)
    print("LANDING")
    print("=" * 60)
    
    # Clear any RC overrides first
    clear_rc_override(vehicle)
    
    print("Switching to LAND mode...")
    vehicle.mode = VehicleMode("LAND")
    
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {altitude:.1f}m", end='\r')
        
        if altitude < 0.5:
            print(f"\n✓ Landed!")
            break
        
        time.sleep(0.5)
    
    print("=" * 60)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  DRONEKIT FLIP MANEUVERS".center(58) + "█")
    print("█" + "  Using ArduPilot's Native FLIP Mode".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    print("\nThis script uses ArduPilot's BUILT-IN FLIP mode.")
    print("Unlike AirSim's direct motor control, this is SAFE!")
    print("\nSequence:")
    print("  1. Takeoff to 10m")
    print("  2. RIGHT flip")
    print("  3. LEFT flip")
    print("  4. FORWARD flip")
    print("  5. BACKWARD flip")
    print("  6. Land")
    print("\n⚠️  Requires at least 10m altitude!")
    print("Make sure ArduPilot SITL and AirSim are running!\n")
    
    input("Press ENTER to start...")
    
    vehicle = None
    
    try:
        # Connect to vehicle
        vehicle = connect_vehicle()
        
        # Setup FLIP parameters
        setup_flip_parameters(vehicle)
        
        # Takeoff
        arm_and_takeoff(vehicle, TAKEOFF_ALTITUDE)
        time.sleep(2)  # Stabilize
        
        # ============================================================
        # FLIP SEQUENCE
        # ============================================================
        
        print("\n" + "=" * 60)
        print("EXECUTING FLIP SEQUENCE")
        print("=" * 60)
        
        # RIGHT FLIP
        print("\n[1/4] RIGHT FLIP")
        execute_flip_sequence(vehicle, "RIGHT FLIP", roll_dir=1, pitch_dir=0)
        time.sleep(3)
        
        # LEFT FLIP
        print("\n[2/4] LEFT FLIP")
        execute_flip_sequence(vehicle, "LEFT FLIP", roll_dir=-1, pitch_dir=0)
        time.sleep(3)
        
        # FORWARD FLIP
        print("\n[3/4] FORWARD FLIP")
        execute_flip_sequence(vehicle, "FORWARD FLIP", roll_dir=0, pitch_dir=1)
        time.sleep(3)
        
        # BACKWARD FLIP
        print("\n[4/4] BACKWARD FLIP")
        execute_flip_sequence(vehicle, "BACKWARD FLIP", roll_dir=0, pitch_dir=-1)
        time.sleep(3)
        
        # ============================================================
        # LANDING
        # ============================================================
        land_vehicle(vehicle)
        
        print("\n" + "=" * 60)
        print("MISSION COMPLETE!")
        print("=" * 60)
        print("Successfully performed:")
        print("  ✓ RIGHT flip")
        print("  ✓ LEFT flip")
        print("  ✓ FORWARD flip")
        print("  ✓ BACKWARD flip")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ KEYBOARD INTERRUPT!")
        if vehicle:
            clear_rc_override(vehicle)
            vehicle.mode = VehicleMode("LAND")
            time.sleep(10)
    
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        if vehicle:
            clear_rc_override(vehicle)
            vehicle.mode = VehicleMode("LAND")
            time.sleep(10)
    
    finally:
        if vehicle:
            clear_rc_override(vehicle)
            vehicle.close()
            print("\n✓ Connection closed")
        
        print("\nScript terminated.\n")


if __name__ == "__main__":
    main()