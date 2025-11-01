import airsim
import numpy as np
import cv2
import time

def main():
    # === CONNECT TO AIRSIM ===
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ Connected and armed")

    # === TAKEOFF ===
    print("\n[1/4] Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(-2, 2).join()
    print("🚁 Drone hovering at 5 meters altitude")
    time.sleep(1)

    # === CONTROL LOOP ===
    print("\n[2/4] Starting red object tracking with bounding box visualization...")
    Kp = 0.0025          # Proportional gain for yaw control
    min_area = 100        # Minimum area to consider object valid
    detection_lost_counter = 0

    # Camera feed dimensions (for center reference)
    frame_center_x = 640 / 2  # assuming 640x480 resolution
    yaw_rate = 0

    try:
        while True:
            # --- Get image from AirSim camera ---
            raw_image = client.simGetImage("0", airsim.ImageType.Scene)
            if raw_image is None:
                continue

            img1d = np.frombuffer(raw_image, dtype=np.uint8)
            frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # === (1) Grayscale for reference ===
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # === (2) HSV for red color segmentation ===
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Adjusted HSV range for Unreal red
            lower_red1 = np.array([0, 80, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([160, 80, 50])
            upper_red2 = np.array([180, 255, 255])

            # Combine two red masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Clean mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # === (3) Detect contours ===
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area > min_area:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cx = x + w / 2
                    cy = y + h / 2

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                    cv2.putText(frame, "Red Object", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # === (4) Drone yaw control ===
                    error_x = cx - frame_center_x
                    yaw_rate = -Kp * error_x  # negative for correction direction
                    yaw_rate = np.clip(yaw_rate, -0.3, 0.3)  # limit yaw rate

                    # Apply slow movement to align
                    client.moveByVelocityBodyFrameAsync(0, 0, 0, duration=0.1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

                    detection_lost_counter = 0
                else:
                    detection_lost_counter += 1
            else:
                detection_lost_counter += 1

            if detection_lost_counter > 10:
                yaw_rate = 0  # stop yaw when no detection
                detection_lost_counter = 0

            # === (5) Visualization ===
            red_overlay = cv2.bitwise_and(frame, frame, mask=mask)
            combined_view = cv2.addWeighted(frame, 0.7, red_overlay, 1.0, 0)

            cv2.imshow("Original with Bounding Box", frame)
            cv2.imshow("HSV Mask (Red Detection)", mask)
            cv2.imshow("Red Object Highlight", combined_view)
            cv2.imshow("Grayscale", gray)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")

    finally:
        print("\n[3/4] Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()
        print("✅ Drone disarmed and API control released")
        print("\n[4/4] Mission Complete!")

if __name__ == "__main__":
    main()
