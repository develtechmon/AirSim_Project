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
        roll_rate = 16.0      # rad/s for roll flips
        pitch_rate = 16.0     # rad/s for pitch flips
        throttle = 0.72
        recovery_thrust = 0.9
        flip_duration = (2 * math.pi) / roll_rate

        # ---------- RIGHT FLIP ----------
        print("\n[2/12] Executing RIGHT flip...")
        client.moveByAngleRatesThrottleAsync(roll_rate, 0, 0, throttle, flip_duration).join()
        print("✓ Right flip complete")

        print("\n[3/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[4/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        time.sleep(2)

        # ---------- LEFT FLIP ----------
        print("\n[5/12] Executing LEFT flip...")
        client.moveByAngleRatesThrottleAsync(-roll_rate, 0, 0, throttle, flip_duration).join()
        print("✓ Left flip complete")

        print("\n[6/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[7/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        time.sleep(2)

        # ---------- FRONT FLIP ----------
        print("\n[8/12] Executing FRONT flip...")
        client.moveByAngleRatesThrottleAsync(0, pitch_rate, 0, throttle, flip_duration).join()
        print("✓ Front flip complete")

        print("\n[9/12] Recovery thrust upward...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()

        print("\n[10/12] Returning to 5 m hover altitude...")
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        time.sleep(2)

        # ---------- BACK FLIP ----------
        print("\n[11/12] Executing BACK flip...")
        client.moveByAngleRatesThrottleAsync(0, -pitch_rate, 0, throttle, flip_duration).join()
        print("✓ Back flip complete")

        # Final recovery
        print("\n[12/12] Final recovery and landing...")
        client.moveByAngleRatesThrottleAsync(0, 0, 0, recovery_thrust, 0.7).join()
        client.moveToZAsync(-5, 2.5).join()
        client.hoverAsync().join()
        time.sleep(2)

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
