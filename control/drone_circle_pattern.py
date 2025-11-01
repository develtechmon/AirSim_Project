import airsim
import math
import time

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
        # Takeoff
        client.takeoffAsync().join()
        client.moveToZAsync(-5, velocity=2).join()
        time.sleep(1)

        # Get center position
        pose = client.simGetVehiclePose()
        center_x, center_y = pose.position.x_val, pose.position.y_val

        # Fly a circle
        radius = 5
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            x = center_x + radius * math.cos(rad)
            y = center_y + radius * math.sin(rad)
            client.moveToPositionAsync(x, y, -5, velocity=2).join()
            time.sleep(0.1)
        # Land
        client.landAsync().join()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()