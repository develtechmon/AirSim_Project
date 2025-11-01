import airsim
import time
import random

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Enable visual weather
client.simEnableWeather(True)

# Scenario 1: Heavy rainstorm with strong wind
print("Simulating heavy rainstorm...")
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.9)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.3)
client.simSetWind(airsim.Vector3r(15, 8, -3))

client.takeoffAsync().join()
client.moveToPositionAsync(0, 50, -20, 5).join()
time.sleep(20)

# Scenario 2: Blizzard conditions
print("Simulating blizzard...")
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 1.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.7)
client.simSetWind(airsim.Vector3r(20, 15, -5))

time.sleep(20)

# Scenario 3: Dust storm with turbulence
print("Simulating dust storm...")
client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 1.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.5)

# Simulate turbulent wind by changing it randomly
for i in range(20):
    turbulent_wind = airsim.Vector3r(
        15 + random.uniform(-8, 8),   # Base wind + turbulence
        10 + random.uniform(-5, 5),
        -3 + random.uniform(-2, 2)
    )
    client.simSetWind(turbulent_wind)
    time.sleep(1)

# Return to calm
client.simSetWind(airsim.Vector3r(0, 0, 0))
client.landAsync().join()