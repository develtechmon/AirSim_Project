import airsim
import numpy as np
import cv2
import time
from pynput import keyboard

# === PID Controller Class ===
class PIDController:
    """PID controller for smooth control"""
    def __init__(self, Kp, Ki, Kd, output_limits=(-1, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def update(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()
            
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.01
            
        # Proportional
        P = self.Kp * error
        
        # Integral (with anti-windup)
        self.integral += error * dt
        max_integral = abs(self.output_limits[1] - self.output_limits[0]) / 2
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.Ki * self.integral
        
        # Derivative
        derivative = (error - self.previous_error) / dt
        D = self.Kd * derivative
        
        # Total output
        output = P + I + D
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()


# === Global Keyboard Variables ===
key_pressed = set()

def on_press(key):
    try:
        key_pressed.add(key.char.lower())
    except:
        pass

def on_release(key):
    try:
        key_pressed.remove(key.char.lower())
    except:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def detect_red_object(frame):
    """Detect red object and return center position + visualization frames"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Red color detection
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Store original mask before morphological operations
    mask_raw = mask.copy()
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    
    # Create red-only view
    red_only = cv2.bitwise_and(frame, frame, mask=mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, 0, None, hsv, gray, mask, red_only
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    if area < 200:
        return None, None, 0, None, hsv, gray, mask, red_only
    
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w / 2
    cy = y + h / 2
    
    return cx, cy, area, (x, y, w, h), hsv, gray, mask, red_only


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ Connected and armed")

    print("🚁 Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(-3, velocity=1).join()
    print("✅ Hovering at 3m")

    # Get actual frame dimensions from first captured frame
    print("📐 Getting camera resolution...")
    raw_image = client.simGetImage("0", airsim.ImageType.Scene)
    img1d = np.frombuffer(raw_image, dtype=np.uint8)
    test_frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
    frame_height, frame_width = test_frame.shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    print(f"✅ Camera: {frame_width}x{frame_height}, Center: ({frame_center_x:.1f}, {frame_center_y:.1f})")

    # === AGGRESSIVE PID - PRIORITIZE CENTERING ===
    pid_yaw = PIDController(
        Kp=0.20,      # MUCH more aggressive
        Ki=0.01,      # Stronger integral to eliminate steady-state error
        Kd=0.10,      # Moderate damping
        output_limits=(-45, 45)  # Allow full rotation speed
    )
    
    # DISABLE forward movement initially - focus on centering first
    pid_forward = PIDController(
        Kp=0.0005,    # Increased from 0.00005 - faster approach
        Ki=0.00002,   # Slightly increased
        Kd=0.0002,
        output_limits=(-0.8, 0.8)  # Increased from 0.1 - much faster forward movement
    )
    
    pid_altitude = PIDController(
        Kp=0.0015,
        Ki=0.0003,
        Kd=0.002,
        output_limits=(-0.3, 0.3)
    )

    target_area = 8000
    detection_lost_counter = 0
    max_lost_frames = 30
    
    # MINIMAL smoothing - more responsive
    smoothed_cx = None
    smoothed_cy = None
    alpha = 0.7  # Higher = less smoothing, more responsive

    print("🎯 AGGRESSIVE tracking mode!")
    print("Priority: CENTER OBJECT FIRST, then approach")
    print("Manual controls: W/S/A/D/Q/E/U/I, ESC to exit")

    try:
        while True:
            current_time = time.time()
            
            raw_image = client.simGetImage("0", airsim.ImageType.Scene)
            if raw_image is None:
                continue

            img1d = np.frombuffer(raw_image, dtype=np.uint8)
            frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            cx, cy, area, bbox, hsv_view, gray_view, mask_view, red_only_view = detect_red_object(frame)

            # Manual control override
            vx = vy = vz = 0
            manual_yaw = 0
            manual_control = False
            speed = 1.0

            if 'w' in key_pressed: 
                vx = speed
                manual_control = True
            if 's' in key_pressed: 
                vx = -speed
                manual_control = True
            if 'a' in key_pressed: 
                vy = -speed
                manual_control = True
            if 'd' in key_pressed: 
                vy = speed
                manual_control = True
            if 'u' in key_pressed: 
                vz = -0.5
                manual_control = True
            if 'i' in key_pressed: 
                vz = 0.5
                manual_control = True
            if 'q' in key_pressed: 
                manual_yaw = -30
                manual_control = True
            if 'e' in key_pressed: 
                manual_yaw = 30
                manual_control = True

            if manual_control:
                client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, 
                    duration=1,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=manual_yaw)
                )
                status_text = "🕹️ MANUAL CONTROL"
                pid_yaw.reset()
                pid_forward.reset()
                pid_altitude.reset()
                detection_lost_counter = 0
                smoothed_cx = None
                smoothed_cy = None
                
            elif cx is not None:
                detection_lost_counter = 0
                
                # Light smoothing
                if smoothed_cx is None:
                    smoothed_cx = cx
                    smoothed_cy = cy
                else:
                    smoothed_cx = alpha * cx + (1 - alpha) * smoothed_cx
                    smoothed_cy = alpha * cy + (1 - alpha) * smoothed_cy
                
                # Use smoothed values
                cx = smoothed_cx
                cy = smoothed_cy
                
                # Draw visualization
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 7, (0, 0, 255), -1)
                cv2.circle(frame, (int(frame_center_x), int(frame_center_y)), 7, (255, 0, 0), -1)
                
                # Draw line showing error
                cv2.line(frame, (int(cx), int(cy)), 
                        (int(frame_center_x), int(frame_center_y)), (0, 255, 255), 2)
                
                # Calculate errors
                error_x = cx - frame_center_x
                error_y = cy - frame_center_y
                error_area = target_area - area
                
                # VERY SMALL DEADZONE - almost none!
                deadzone_x = 5  # Only 5 pixels!
                deadzone_y = 30
                
                if abs(error_x) < deadzone_x:
                    error_x = 0
                if abs(error_y) < deadzone_y:
                    error_y = 0
                
                # PID outputs
                yaw_rate = pid_yaw.update(error_x, current_time)  # FIXED: removed negative sign
                
                # FORCE minimum yaw rate if error is large (prevent stuck at zero)
                if abs(error_x) > 20 and abs(yaw_rate) < 2:
                    yaw_rate = -5.0 if error_x < 0 else 5.0  # FIXED: direction matches error
                    print(f"⚠️  FORCING minimum yaw rate: {yaw_rate}°/s")
                
                # Ensure yaw_rate is within bounds
                yaw_rate = np.clip(yaw_rate, -45, 45)
                
                # Only move forward if well-centered (error < 50px)
                if abs(error_x) < 50:
                    vx = pid_forward.update(error_area, current_time)
                else:
                    vx = 0  # Don't approach until centered!
                
                vz = pid_altitude.update(error_y, current_time)
                
                # DEBUG OUTPUT
                print(f"error_x:{error_x:7.1f}px | yaw_rate:{yaw_rate:7.2f}°/s | vx:{vx:6.2f} | area:{int(area):5d}")
                
                # Verify yaw_rate is non-zero for large errors
                if abs(error_x) > 30 and abs(yaw_rate) < 1:
                    print(f"  ⚠️  WARNING: Large error but small yaw_rate! Check PID!")
                
                # Send control
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                print(f"  → Sending: vx={vx:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}°/s")
                
                client.moveByVelocityBodyFrameAsync(
                    vx, 0, vz,
                    duration=1,
                    yaw_mode=yaw_mode
                )
                
                # Status
                if abs(error_x) < 50:
                    status_text = f"🎯 CENTERED | Approaching"
                else:
                    status_text = f"🔄 CENTERING | Yaw:{yaw_rate:.1f}°/s"
                
                # Debug info on screen
                cv2.putText(frame, f"Error X: {error_x:.1f}px", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Yaw Rate: {yaw_rate:.1f} deg/s", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Forward: {vx:.2f} m/s", (20, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Area: {int(area)} (Target: {target_area})", (20, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            else:
                detection_lost_counter += 1
                smoothed_cx = None
                smoothed_cy = None
                
                if detection_lost_counter < max_lost_frames:
                    client.hoverAsync()
                    status_text = f"⏸️ SEARCHING... ({detection_lost_counter}/{max_lost_frames})"
                else:
                    client.hoverAsync()
                    status_text = "❌ TARGET LOST"
                
                pid_yaw.reset()
                pid_forward.reset()
                pid_altitude.reset()

            # Display status
            cv2.putText(frame, status_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Center crosshair
            cv2.line(frame, (int(frame_center_x) - 30, int(frame_center_y)), 
                    (int(frame_center_x) + 30, int(frame_center_y)), (255, 0, 0), 2)
            cv2.line(frame, (int(frame_center_x), int(frame_center_y) - 30), 
                    (int(frame_center_x), int(frame_center_y) + 30), (255, 0, 0), 2)
            
            cv2.imshow("Red Object Tracking - AGGRESSIVE MODE", frame)
            
            # Additional visualization windows
            cv2.imshow("HSV View", hsv_view)
            cv2.imshow("Grayscale", gray_view)
            cv2.imshow("Red Mask", mask_view)
            cv2.imshow("Red Objects Only", red_only_view)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        
    finally:
        print("\n🛬 Landing...")
        try:
            client.landAsync().join()
        except:
            print("Landing completed")
        
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()
        listener.stop()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    main()