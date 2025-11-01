import airsim
import time

"""
AirSim Coordinate System (NED - North East Down):
- X axis: Forward (positive) / Backward (negative)
- Y axis: Right (positive) / Left (negative)
- Z axis: Down (positive) / Up (negative)

So:
- Move UP: z = negative
- Move DOWN: z = positive
- Move FORWARD: x = positive
- Move BACKWARD: x = negative
- Move RIGHT: y = positive
- Move LEFT: y = negative
"""
target_distance = 10

def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim!")

    # Enable API control
    client.enableApiControl(True)
    print("✓ API control enabled")
    
    # Arm the drone
    client.armDisarm(True)
    print("✓ Drone armed")
    
    try:
        # Take off
        print("\n[1/6] Taking off...")
        client.takeoffAsync().join()
        print("✓ Takeoff complete")
        
        # Move to 5 meters altitude
        print("\n[2/6] Moving to 5 meters altitude...")
        client.moveToZAsync(z=-5, velocity=2).join()
        print("✓ At 5 meters altitude")
        time.sleep(1)
        
        # Get current position
        pose = client.simGetVehiclePose()
        start_x = pose.position.x_val
        start_y = pose.position.y_val
        print(f"Starting position: X={start_x:.2f}, Y={start_y:.2f}")

        # Move LEFT target_distance meter
        print("\n[3/6] Moving LEFT target_distance meter...")
        target_y = start_y - target_distance  # Left is negative Y
        client.moveToPositionAsync(start_x, target_y, -5, velocity=1).join()
        print("✓ Moved left")
        time.sleep(1)
        
        # Move RIGHT target_distance meter (back to start, then 1m more right)
        print("\n[4/6] Moving RIGHT target_distance meter...")
        target_y = start_y + target_distance  # Right is positive Y
        client.moveToPositionAsync(start_x, target_y, -5, velocity=1).join()
        print("✓ Moved right")
        time.sleep(1)
        
        # Move BACKWARD 2 meter
        print("\n[5/6] Moving BACKWARD target_distance meter...")
        target_x = start_x - target_distance  # Backward is negative X
        client.moveToPositionAsync(target_x, target_y, -5, velocity=1).join()
        print("✓ Moved backward")
        time.sleep(1)

        # Land
        print("\n[6/6] Landing...")
        client.landAsync().join()
        print("✓ Landed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Emergency landing...")
        client.landAsync().join()
    
    finally:
        # Cleanup
        client.armDisarm(False)
        print("\n✓ Drone disarmed")
        client.enableApiControl(False)
        print("✓ API control disabled")
        print("\nFlight complete! You can now use RC control again.")
        
if __name__ == "__main__":
    main()