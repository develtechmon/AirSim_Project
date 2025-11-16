from dronekit import connect

# Connect to ArduPilot
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)

# Check connection
print(f"Mode: {vehicle.mode.name}")
print(f"GPS: {vehicle.gps_0.fix_type}")
print(f"Battery: {vehicle.battery.level}%")

# Close
vehicle.close()