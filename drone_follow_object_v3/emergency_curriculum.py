"""
COORDINATE SYSTEM VERIFICATION TEST

This script tests if the coordinate system is working correctly.
It will move the drone in each axis and verify the movement matches expectations.

AirSim Coordinate System (NED - North-East-Down):
- X axis: Lateral (positive = right/east, negative = left/west)
- Y axis: Forward/Backward (positive = forward/north, negative = backward/south)
- Z axis: Vertical (positive = down, negative = up)
"""

import airsim
import numpy as np
import time


def test_coordinate_system():
    print("="*70)
    print("COORDINATE SYSTEM VERIFICATION TEST")
    print("="*70)
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("\n✓ Connected to AirSim")
    
    # Take off
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Move to starting position
    print("Moving to starting position (0, 0, -10)...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Get initial position
    state = client.getMultirotorState()
    start_pos = state.kinematics_estimated.position
    print(f"Starting position: X={start_pos.x_val:.2f}, Y={start_pos.y_val:.2f}, Z={start_pos.z_val:.2f}")
    
    print("\n" + "="*70)
    print("TESTING EACH AXIS")
    print("="*70)
    
    # Test X axis (lateral - right/left)
    print("\n--- Test 1: X Axis (Lateral Movement) ---")
    print("Sending command: vx=+5.0 (should move RIGHT/EAST)")
    client.moveByVelocityAsync(5.0, 0.0, 0.0, duration=2.0).join()
    time.sleep(0.5)
    
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    delta_x = pos.x_val - start_pos.x_val
    delta_y = pos.y_val - start_pos.y_val
    delta_z = pos.z_val - start_pos.z_val
    
    print(f"Result: X changed by {delta_x:.2f}m, Y changed by {delta_y:.2f}m, Z changed by {delta_z:.2f}m")
    
    if delta_x > 5:
        print("✅ PASS: Moved right as expected")
    else:
        print(f"❌ FAIL: Expected X to increase significantly, but only changed by {delta_x:.2f}m")
    
    # Return to start
    print("\nReturning to start...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Test Y axis (forward/backward)
    print("\n--- Test 2: Y Axis (Forward/Backward Movement) ---")
    print("Sending command: vy=+5.0 (should move FORWARD/NORTH)")
    client.moveByVelocityAsync(0.0, 5.0, 0.0, duration=2.0).join()
    time.sleep(0.5)
    
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    delta_x = pos.x_val - start_pos.x_val
    delta_y = pos.y_val - start_pos.y_val
    delta_z = pos.z_val - start_pos.z_val
    
    print(f"Result: X changed by {delta_x:.2f}m, Y changed by {delta_y:.2f}m, Z changed by {delta_z:.2f}m")
    
    if delta_y > 5:
        print("✅ PASS: Moved forward as expected")
    else:
        print(f"❌ FAIL: Expected Y to increase significantly, but only changed by {delta_y:.2f}m")
    
    # Return to start
    print("\nReturning to start...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Test Z axis (up/down)
    print("\n--- Test 3: Z Axis (Vertical Movement) ---")
    print("Sending command: vz=+3.0 (should move DOWN in NED coordinates)")
    client.moveByVelocityAsync(0.0, 0.0, 3.0, duration=2.0).join()
    time.sleep(0.5)
    
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    delta_x = pos.x_val - start_pos.x_val
    delta_y = pos.y_val - start_pos.y_val
    delta_z = pos.z_val - start_pos.z_val
    
    print(f"Result: X changed by {delta_x:.2f}m, Y changed by {delta_y:.2f}m, Z changed by {delta_z:.2f}m")
    
    if delta_z > 3:
        print("✅ PASS: Moved down as expected (Z increased in NED)")
    else:
        print(f"❌ FAIL: Expected Z to increase, but only changed by {delta_z:.2f}m")
    
    # Return to start
    print("\nReturning to start...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Test diagonal movement
    print("\n--- Test 4: Diagonal Movement (Combined) ---")
    print("Sending command: vx=+3.0, vy=+3.0, vz=0.0")
    print("(should move diagonally: RIGHT + FORWARD)")
    client.moveByVelocityAsync(3.0, 3.0, 0.0, duration=2.0).join()
    time.sleep(0.5)
    
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    delta_x = pos.x_val - start_pos.x_val
    delta_y = pos.y_val - start_pos.y_val
    delta_z = pos.z_val - start_pos.z_val
    
    print(f"Result: X changed by {delta_x:.2f}m, Y changed by {delta_y:.2f}m, Z changed by {delta_z:.2f}m")
    
    if delta_x > 3 and delta_y > 3:
        print("✅ PASS: Moved diagonally as expected")
    else:
        print(f"❌ FAIL: Expected both X and Y to increase")
    
    # Land
    print("\n" + "="*70)
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ Test complete!")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nIf all tests passed:")
    print("  ✅ Coordinate system is working correctly")
    print("  ✅ Drone responds to velocity commands")
    print("  ✅ Environment should work for training")
    print("\nIf any tests failed:")
    print("  ❌ There may be a coordinate system mismatch")
    print("  ❌ Check AirSim settings or drone configuration")


def test_target_chase_logic():
    """Test if the chase logic works correctly"""
    print("\n" + "="*70)
    print("CHASE LOGIC VERIFICATION")
    print("="*70)
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Take off
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Move to start
    print("Moving to starting position (0, 0, -10)...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Define target position
    target_x, target_y, target_z = 20.0, 15.0, -12.0
    print(f"\nTarget at: ({target_x}, {target_y}, {target_z})")
    
    # Get drone position
    state = client.getMultirotorState()
    drone_pos = state.kinematics_estimated.position
    print(f"Drone at: ({drone_pos.x_val:.2f}, {drone_pos.y_val:.2f}, {drone_pos.z_val:.2f})")
    
    # Calculate relative position (what the observation would be)
    dx = target_x - drone_pos.x_val
    dy = target_y - drone_pos.y_val
    dz = target_z - drone_pos.z_val
    
    print(f"\nRelative position: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
    
    # Calculate direction (normalized)
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    print(f"Distance to target: {distance:.2f}m")
    
    vx = (dx / distance) * 5.0
    vy = (dy / distance) * 5.0
    vz = (dz / distance) * 5.0
    
    print(f"\nCalculated velocity command: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
    print("This should move the drone toward the target...")
    
    # Execute for 3 seconds
    print("\nExecuting movement for 3 seconds...")
    for i in range(3):
        client.moveByVelocityAsync(vx, vy, vz, duration=1.0).join()
        
        # Check new position
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        new_distance = np.sqrt(
            (target_x - pos.x_val)**2 +
            (target_y - pos.y_val)**2 +
            (target_z - pos.z_val)**2
        )
        
        print(f"  Step {i+1}: Position=({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}), Distance={new_distance:.2f}m")
    
    if new_distance < distance:
        print(f"\n✅ SUCCESS: Distance decreased from {distance:.2f}m to {new_distance:.2f}m")
        print("   Chase logic is working correctly!")
    else:
        print(f"\n❌ FAILURE: Distance did not decrease!")
        print(f"   Started at {distance:.2f}m, ended at {new_distance:.2f}m")
        print("   There may be a coordinate system issue")
    
    # Land
    print("\nLanding...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ Chase logic test complete!")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         AirSim Coordinate System Verification                     ║
╚═══════════════════════════════════════════════════════════════════╝

This will test:
1. X axis (lateral: right/left)
2. Y axis (forward/backward)
3. Z axis (vertical: up/down)
4. Chase logic (moving toward target)

Make sure AirSim is running before starting!
    """)
    
    input("Press Enter to start tests...")
    
    try:
        # Test basic axes
        test_coordinate_system()
        
        print("\n" * 3)
        input("Press Enter to test chase logic...")
        
        # Test chase logic
        test_target_chase_logic()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure AirSim is running and the drone is in a flyable environment!")