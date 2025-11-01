import airsim
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()

# Get IMU data
imu_data = client.getImuData()
# Returns: angular_velocity (x,y,z rad/s), linear_acceleration (x,y,z m/s²), orientation (quaternion)
print(imu_data)

# Get kinematics for expected state
state = client.getMultirotorState()
# Returns: kinematics_estimated (position, velocity, acceleration, orientation)
print(state)

# Get collision data
collision_info = client.simGetCollisionInfo()
# Returns: has_collided, collision_position, surface_normal, penetration_depth
print(collision_info)