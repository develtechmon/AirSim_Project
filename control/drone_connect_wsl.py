# For this one i'm using python 3.11.9 in WSL
import airsim
import time

client = airsim.MultirotorClient(ip="172.23.128.1") # This is referring to windows IP address "IPv4 Address"
client.confirmConnection()
print("Connected to AirSim drone simulator.")
