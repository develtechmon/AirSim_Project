import airsim
import numpy as np
import cv2
import time

# Connect
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# Move forward (in NED: +X is forward)
print("Moving forward...")
client.moveByVelocityAsync(3, 0, 0, 3).join()  # 3 m/s forward for 3 seconds
time.sleep(1)

# Get drone state
print("\n=== Drone State ===")
state = client.getMultirotorState()
drone_pos = state.kinematics_estimated.position
print(f"Position: X={drone_pos.x_val:.2f}, Y={drone_pos.y_val:.2f}, Z={drone_pos.z_val:.2f}")

# Get gate positions (correct API call)
print("\n=== Gate Positions ===")
gates = {}
gate_names = ['Gate02', 'Gate03', 'Gate04', 'Gate05', 'Gate06', 'Gate07', 'Gate08', 'Gate09', 'Gate10_21', 'Gate11_23']

for gate_name in gate_names:
    try:
        # Correct call - only one parameter
        pose = client.simGetObjectPose(gate_name)
        gates[gate_name] = pose.position
        
        # Calculate distance
        dx = pose.position.x_val - drone_pos.x_val
        dy = pose.position.y_val - drone_pos.y_val
        dz = pose.position.z_val - drone_pos.z_val
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"✓ {gate_name}: pos=({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f}), dist={distance:.1f}m")
    except Exception as e:
        print(f"✗ {gate_name}: {e}")

# Get camera image (correct API call - simpler version)
print("\n=== Camera Feed ===")
try:
    # Method 1: Try simple call
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    
    if responses and responses[0].height > 0:
        # Convert to numpy
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        
        print(f"✓ Camera resolution: {img_rgb.shape}")
        
        # Save image
        cv2.imwrite("drone_view.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print("✓ Saved drone_view.png")
        
        # Show a small preview of what drone sees
        print(f"✓ Image stats: min={img_rgb.min()}, max={img_rgb.max()}, mean={img_rgb.mean():.1f}")
    else:
        print("✗ Empty image response")
        
except Exception as e:
    print(f"✗ Camera failed: {e}")
    print("Trying alternate method...")
    
    try:
        # Method 2: Compressed PNG
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True)
        ])
        if responses:
            print("✓ Compressed image method worked")
            # Save directly
            airsim.write_file("drone_view.png", responses[0].image_data_uint8)
    except Exception as e2:
        print(f"✗ Also failed: {e2}")

# Land
print("\n=== Landing ===")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("\n" + "="*60)
print("SUMMARY:")
print(f"✓ Found {len(gates)} gates with API access")
print("✓ Can calculate distance/direction to each gate")
print("✓ This means we can build SMART rewards!")
print("="*60)
print("\nNext: Upload drone_view.png so I can see what the camera sees")