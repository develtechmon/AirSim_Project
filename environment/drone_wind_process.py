import airsim
import time
import random

client = airsim.MultirotorClient()
client.confirmConnection()

print("Wind process started")

while True:
    # Random wind between -5 and 5 m/s (simple test)
    wind_x = random.uniform(-20, 20)
    wind_y = random.uniform(-20, 20)
    wind_z = 0  # keep zero for now for easier observation
    
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 1.0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 50)
    client.simSetWind(airsim.Vector3r(wind_x, wind_y, wind_z))
    print(f"Wind set to: {wind_x:.2f}, {wind_y:.2f}, {wind_z:.2f}")

    time.sleep(3)  # change every 3 seconds
