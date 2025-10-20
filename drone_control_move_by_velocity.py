import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim drone simulator.")

# Enable API control (needed to control via code)
client.enableApiControl(True)
print("API control enabled")

# Arm the drone
client.armDisarm(True)
print("Drone armed")

# Take off to 5 meters
print("Taking off...")
client.takeoffAsync().join()  # Takeoff (goes to ~3m by default)
print("Takeoff complete, now moving to 5 meters altitude")

# Move to exactly 5 meters altitude (z = -5 in NED coordinates)
# NED: North-East-Down, so negative Z means UP
client.moveToZAsync(-5, velocity=2).join()
print("Reached 5 meters altitude")

# Hover for 2 seconds
print("Hovering for 2 seconds...")
time.sleep(2)

# Move LEFT 1 meter (negative Y in NED coordinates)
print("Moving LEFT 1 meter...")
client.moveByVelocityAsync(vx=0, vy=-2, vz=0, duration=1).join()
print("Moved left")
time.sleep(1)

# Move RIGHT 1 meter (positive Y in NED coordinates)
print("Moving RIGHT 1 meter...")
client.moveByVelocityAsync(vx=0, vy=2, vz=0, duration=1).join()
print("Moved right")
time.sleep(1)

# Move BACKWARD 1 meter (negative X in NED coordinates)
print("Moving BACKWARD 1 meter...")
client.moveByVelocityAsync(vx=-2, vy=0, vz=0, duration=1).join()
print("Moved backward")
time.sleep(1)

# Hover for 2 seconds before landing
print("Hovering before landing...")
time.sleep(2)

# Land
print("Landing...")
client.landAsync().join()
print("Landed!")

# Disarm
client.armDisarm(False)
print("Drone disarmed")

# Disable API control
client.enableApiControl(False)
print("API control disabled - you can now use RC again")
