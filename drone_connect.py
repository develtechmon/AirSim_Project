import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim drone simulator.")