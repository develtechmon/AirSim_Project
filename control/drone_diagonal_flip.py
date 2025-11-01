import airsim
import time
import math

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim!")

    client.enableApiControl(True)
    client.armDisarm(True)
    print("✓ Drone armed and API control enabled")

    try:
        # Step 1: Takeoff to safe altitude
        print("\n[1/12] Taking off to 8 meters...")
        client.takeoffAsync().join()
        client.moveToZAsync(-8, 2).join()
        print("✓ Hovering at 8 m altitude")
        time.sleep(1)

        # Common flip settings
        base_rate = 16.0
        throttle = 0.72
        recovery_thrust = 0.9
        flip_duration = (2 * math.pi) / base_rate
        
        # Diagonal flip rate (divide by √2 for 45° angle)
        diagonal_rate = base_rate / math.sqrt(2)  # ~11.3 rad/s

        # ---------- DIAGONAL FRONT-RIGHT FLIP ----------
        print("\n[2/12] Executing DIAGONAL FRONT-RIGHT flip...")
        client.moveByAngleRatesThrottleAsync(
            diagonal_rate,   # +Roll (right)
            diagonal_rate,   # +Pitch (forward)
            0, 
            throttle, 
            flip_duration
        ).join()
        print("✓ Diagonal front-right flip complete")

        print("\n[3/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[4/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-8, 2.5).join()
        client.hoverAsync().join()
        time.sleep(5)

        # ---------- DIAGONAL FRONT-LEFT FLIP ----------
        print("\n[5/12] Executing DIAGONAL FRONT-LEFT flip...")
        client.moveByAngleRatesThrottleAsync(
            -diagonal_rate,  # -Roll (left)
            diagonal_rate,   # +Pitch (forward)
            0, 
            throttle, 
            flip_duration
        ).join()
        print("✓ Diagonal front-left flip complete")

        print("\n[6/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[7/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-8, 2.5).join()
        client.hoverAsync().join()
        time.sleep(5)

        # ---------- DIAGONAL BACK-RIGHT FLIP ----------
        print("\n[8/12] Executing DIAGONAL BACK-RIGHT flip...")
        client.moveByAngleRatesThrottleAsync(
            diagonal_rate,   # +Roll (right)
            -diagonal_rate,  # -Pitch (backward)
            0, 
            throttle, 
            flip_duration
        ).join()
        print("✓ Diagonal back-right flip complete")

        print("\n[9/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[10/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-8, 2.5).join()
        client.hoverAsync().join()
        time.sleep(5)

        # ---------- DIAGONAL BACK-LEFT FLIP ----------
        print("\n[11/12] Executing DIAGONAL BACK-LEFT flip...")
        client.moveByAngleRatesThrottleAsync(
            -diagonal_rate,  # -Roll (left)
            -diagonal_rate,  # -Pitch (backward)
            0, 
            throttle, 
            flip_duration
        ).join()
        print("✓ Diagonal back-left flip complete")

        # Final recovery
        print("\n[12/12] Final recovery and landing...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()
        client.moveToZAsync(-8, 2.5).join()
        client.hoverAsync().join()
        time.sleep(5)

        client.landAsync().join()
        print("✓ Landed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        client.landAsync().join()

    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        print("\n✓ Drone disarmed and API control disabled")
        print("Flight complete!")

if __name__ == "__main__":
    main()