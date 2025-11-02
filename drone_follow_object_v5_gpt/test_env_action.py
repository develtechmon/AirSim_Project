"""
Manual Action Test for AirSimSphereChaseEnv (Z-locked)
-----------------------------------------------------
Use this script to verify your environment logic before PPO training.
"""

import numpy as np
import time
from airsim_sphere_env_zlocked import AirSimSphereChaseEnv

if __name__ == "__main__":
    env = AirSimSphereChaseEnv(visualize=True)
    obs, info = env.reset()

    print("\n✅ Environment ready.")
    print("Target:", info['target'])
    print("Observation shape:", obs.shape)

    print("\nControl instructions:")
    print("  w = forward (+Y)")
    print("  s = backward (-Y)")
    print("  a = left (-X)")
    print("  d = right (+X)")
    print("  q = quit")

    while True:
        key = input("\nCommand (w/s/a/d/q): ").strip().lower()
        if key == "q":
            break
        if key == "w":
            action = np.array([0.0, 1.0])
        elif key == "s":
            action = np.array([0.0, -1.0])
        elif key == "a":
            action = np.array([-1.0, 0.0])
        elif key == "d":
            action = np.array([1.0, 0.0])
        else:
            action = np.array([0.0, 0.0])

        obs, reward, done, trunc, info = env.step(action)
        print(f"→ DistXY={info['distance_xy']:.2f} | AltDiff={info['alt_diff']:.2f} | Reward={reward:.3f}")

        time.sleep(0.1)
        if done:
            print("🎯 Target reached! Respawning new target.")
            time.sleep(1)

    env.close()
    print("✅ Test finished, drone landed.")
