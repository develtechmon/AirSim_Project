# force_chase_sphere_v3.py
"""
FORCE-CHASE CONTROLLER (No RL) — msgpack-safe
- Forces the drone to chase a red sphere.
- Locks altitude to target (moveByVelocityZAsync).
- Casts ALL numbers passed to AirSim RPC to native floats to avoid
  AttributeError: 'numpy.float32' object has no attribute 'to_msgpack'
"""

import airsim
import numpy as np
import time
import math

# ========= CONFIG (plain Python floats) =========
UPDATE_RATE   = 0.05      # s
MAX_SPEED     = 7.0       # m/s
APPROACH_GAIN = 0.8
STOP_DISTANCE = 1.8       # m
SPHERE_RADIUS = 2.0
TARGET_COLOR  = [1.0, 0.0, 0.0, 1.0]  # RGBA
Z_MIN, Z_MAX  = -25.0, -10.0
DYNAMIC_TARGET = False
DYNAMIC_SPEED  = 0.4      # m/s when DYNAMIC_TARGET=True

# ---------- msgpack-safe float ----------
def pf(x):
    """Return a native Python float (never numpy scalar)."""
    return float(x.item() if isinstance(x, np.generic) else x)

# ---------- Sphere drawing ----------
def create_point_cloud_sphere(client, position, radius=SPHERE_RADIUS, color=TARGET_COLOR):
    """Visualize target sphere using AirSim point cloud (all args cast to float)."""
    num_points = 150
    phi = math.pi * (3.0 - math.sqrt(5.0))

    px, py, pz = map(pf, position)
    r = pf(radius)

    points = []
    for i in range(num_points):
        y_offset = 1 - (i / float(num_points - 1)) * 2
        r_at_y = math.sqrt(max(0.0, 1 - y_offset * y_offset))
        theta = phi * i
        x = math.cos(theta) * r_at_y * r
        y = math.sin(theta) * r_at_y * r
        z = y_offset * r
        points.append(airsim.Vector3r(pf(px + x), pf(py + y), pf(pz + z)))

    client.simPlotPoints(
        points,
        color_rgba=[pf(c) for c in color],
        size=pf(15.0),
        duration=pf(10.0),
        is_persistent=True
    )

# ---------- Target spawn/move ----------
def spawn_random_target(client, r_min=10.0, r_max=30.0, z_min=Z_MIN, z_max=Z_MAX):
    angle = np.random.uniform(0.0, 2.0 * math.pi)
    radius = np.random.uniform(r_min, r_max)
    tx = pf(radius * math.cos(angle))
    ty = pf(radius * math.sin(angle))
    tz = pf(np.random.uniform(z_min, z_max))
    target = np.array([tx, ty, tz], dtype=np.float64)  # internal dtype doesn't matter
    client.simFlushPersistentMarkers()
    create_point_cloud_sphere(client, target, radius=SPHERE_RADIUS)
    return target

def move_target_randomly(target, step_size=0.5):
    d = np.random.uniform(-1, 1, size=3)
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n > 1e-6:
        d = d / n
    return target + d * pf(step_size)

# ---------- Main ----------
def main():
    print("\n=== FORCE CHASE SPHERE v3 (msgpack-safe) ===")

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Takeoff…")
    client.takeoffAsync().join()
    time.sleep(0.5)
    client.moveToZAsync(pf(-10.0), pf(3.0)).join()

    target = spawn_random_target(client)
    print(f"Target @ {target}")

    try:
        while True:
            s = client.getMultirotorState()
            p = s.kinematics_estimated.position
            drone_pos = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float64)

            rel = target - drone_pos
            dist_xy = float(np.linalg.norm(rel[:2]))
            dz = float(target[2] - drone_pos[2])

            # Hit & respawn
            if dist_xy < STOP_DISTANCE:
                print(f"🎯 HIT (dist={dist_xy:.2f}). Respawning…")
                target = spawn_random_target(client)
                continue

            # Optional moving target
            if DYNAMIC_TARGET:
                target = move_target_randomly(target, step_size=DYNAMIC_SPEED * UPDATE_RATE)
                client.simFlushPersistentMarkers()
                create_point_cloud_sphere(client, target, radius=SPHERE_RADIUS)

            # Proportional guidance toward target (XY)
            rel_xy = rel[:2]
            n_xy = np.linalg.norm(rel_xy)
            if n_xy < 1e-6:
                vx, vy = 0.0, 0.0
            else:
                dir_xy = rel_xy / n_xy
                vx = pf(dir_xy[0] * MAX_SPEED * APPROACH_GAIN)
                vy = pf(dir_xy[1] * MAX_SPEED * APPROACH_GAIN)

            z_cmd = pf(target[2])  # lock altitude to target

            # ---- AirSim command (ALL native floats) ----
            client.moveByVelocityZAsync(
                pf(vx), pf(vy), pf(z_cmd), pf(UPDATE_RATE),
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=pf(0.0))
            )

            print(f"DistXY={dist_xy:6.2f} | dZ={dz:6.2f} | Cmd=({vx:+.2f},{vy:+.2f}) | Zcmd={z_cmd:+.2f}")
            time.sleep(pf(UPDATE_RATE))

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try:
            client.hoverAsync().join()
            client.landAsync().join()
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        print("✅ Landed.")
        
if __name__ == "__main__":
    main()
