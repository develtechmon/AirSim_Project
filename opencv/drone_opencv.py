import airsim
import cv2
import numpy as np
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToZAsync(-5, 2).join()

print("📷 Starting camera stream...")

while True:
    # Request image from front camera
    raw_image = client.simGetImage("0", airsim.ImageType.Scene)
    if raw_image is None:
        continue
    
    # Convert to numpy array
    np_arr = np.frombuffer(raw_image, np.uint8)
    img_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Show image
    cv2.imshow("AirSim Camera", img_rgb)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
