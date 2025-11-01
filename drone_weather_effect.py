import airsim

"""
# All available weather types
airsim.WeatherParameter.Rain        # 0 - Rain particles
airsim.WeatherParameter.Roadwetness # 1 - Wet road surface*
airsim.WeatherParameter.Snow        # 2 - Snow particles
airsim.WeatherParameter.RoadSnow    # 3 - Snow on road*
airsim.WeatherParameter.MapleLeaf   # 4 - Falling leaves
airsim.WeatherParameter.RoadLeaf    # 5 - Leaves on road*
airsim.WeatherParameter.Dust        # 6 - Dust particles
airsim.WeatherParameter.Fog         # 7 - Fog effect    
airsim.WeatherParameter.Wind        # 8 - Wind effect (visual only)*
airsim.WeatherParameter.Overcast    # 9 - Overcast sky effect*
"""
# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# CRITICAL: Must enable weather first
client.simEnableWeather(True)

# Set different weather effects (0.0 to 1.0 intensity)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.75)    # Heavy rain
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.50)     # Medium fog
client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.30)    # Light snow
client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0.40)    # Dust storm

# Clear all weather
# client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.0)
# client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.0)
# client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)