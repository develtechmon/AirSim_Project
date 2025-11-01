import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

# Hover at 3 meters
client.moveToZAsync(-3, 1).join()
print("Hovering...")

# Keep script alive (no control after takeoff)
while True:
    state = client.getMultirotorState()
    print(f"Pos: {state.kinematics_estimated.position}")
    time.sleep(1)
