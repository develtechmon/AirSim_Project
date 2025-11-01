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
        # Step 1: Takeoff higher (for flip safety margin)
        print("\n[1/8] Taking off to 8 meters...")
        client.takeoffAsync().join()
        client.moveToZAsync(-8, 2).join()
        print("✓ Hovering at 8 m altitude")
        time.sleep(1)

        # Step 2: Execute RIGHT flip
        print("\n[2/8] Executing RIGHT flip...")
        roll_rate = 16.0      # rad/s (fast spin)
        flip_duration = (2 * math.pi) / roll_rate  # full rotation (2π radians)
        throttle = 0.72       # maintain upward thrust during flip
        client.moveByAngleRatesThrottleAsync(roll_rate, 0, 0, throttle, flip_duration).join()
        print("✓ Right flip complete")

        # Step 3: Recovery thrust
        print("\n[3/8] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, 0.9, 0.7).join()
        print("✓ Recovery thrust applied")

        # Step 4: Return to 5 m altitude and hover
        print("\n[4/8] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        print("✓ Hover stabilized at 5 m")
        time.sleep(2)

        # Step 5: Execute LEFT flip (instead of flight left)
        print("\n[5/8] Executing LEFT flip...")
        roll_rate_left = -16.0    # roll left (negative)
        flip_duration_left = (2 * math.pi) / abs(roll_rate_left)
        throttle_left = 0.72
        client.moveByAngleRatesThrottleAsync(roll_rate_left, 0, 0, throttle_left, flip_duration_left).join()
        print("✓ Left flip complete")

        # Step 6: Recovery thrust again
        print("\n[6/8] Recovery thrust upward after left flip...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, 0.9, 0.7).join()
        print("✓ Recovery thrust applied")

        # Step 7: Return to 5 m altitude and hover
        print("\n[7/8] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        print("✓ Hover stabilized at 5 m")
        time.sleep(3)

        # Step 8: Land
        print("\n[8/8] Landing...")
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
