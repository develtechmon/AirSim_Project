import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Start with calm conditions
client.takeoffAsync().join()

# Simulate increasing wind conditions
# Light wind (5 m/s)
wind = airsim.Vector3r(5, 0, 0)
client.simSetWind(wind)
time.sleep(10)

# Moderate wind (10 m/s crosswind)
wind = airsim.Vector3r(5, 10, 0)
client.simSetWind(wind)
time.sleep(10)

# Strong wind (20 m/s headwind)
wind = airsim.Vector3r(20, 0, 0)
client.simSetWind(wind)
time.sleep(10)

# EXTREME conditions (30 m/s with vertical component)
wind = airsim.Vector3r(25, 15, -5)  # Strong headwind + crosswind + updraft
client.simSetWind(wind)