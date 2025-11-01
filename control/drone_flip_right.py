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
        # Step 1: Takeoff higher (safety margin)
        print("\n[1/6] Taking off to 8 meters...")
        client.takeoffAsync().join()
        client.moveToZAsync(-8, 2).join()
        print("✓ Hovering at 8 m altitude")
        time.sleep(1)

        # Step 2: Hover stabilization
        client.hoverAsync().join()
        time.sleep(1)

        # Step 3: Perform RIGHT flip (fast)
        print("\n[2/6] Executing RIGHT flip...")
        roll_rate = 16.0  # rad/s → ~915°/s (very fast)
        flip_duration = (2 * math.pi) / roll_rate  # full 360° roll
        throttle = 0.72  # moderate lift
        client.moveByAngleRatesThrottleAsync(roll_rate, 0, 0, throttle, flip_duration).join()
        print("✓ Flip executed!")

        # Step 4: Strong recovery thrust right after flip
        print("\n[3/6] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, 0.9, 0.7).join()  # strong boost for 0.7s
        print("✓ Recovery thrust complete")

        # Step 5: Correct back to 5 m hover altitude
        print("\n[4/6] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        time.sleep(5)
        print("✓ Stable hover at 5 m for 5 seconds")

        # Step 6: Land
        print("\n[5/6] Landing...")
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
