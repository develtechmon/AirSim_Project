"""
MINIMAL WORKING EXAMPLE: AirSim Object Spawning

Run this file to see all spawning methods in action!
This is a standalone demo - no training needed.

What this does:
1. Connects to AirSim
2. Takes off
3. Spawns 4 different visual targets using different methods
4. Flies the drone near each one
"""

import airsim
import numpy as np
import time


def create_point_cloud_sphere(client, position, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
    """
    Create a sphere made of points.
    This is the RECOMMENDED method!
    """
    points = []
    num_points = 150
    
    # Fibonacci sphere algorithm
    for i in range(num_points):
        phi = np.pi * (3. - np.sqrt(5.))
        y_offset = 1 - (i / float(num_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y_offset * y_offset)
        theta = phi * i
        
        x = np.cos(theta) * radius_at_y * radius
        y = np.sin(theta) * radius_at_y * radius
        z = y_offset * radius
        
        point = airsim.Vector3r(
            position[0] + x,
            position[1] + y,
            position[2] + z
        )
        points.append(point)
    
    client.simPlotPoints(
        points,
        color_rgba=color,
        size=15.0,
        duration=60.0,
        is_persistent=True
    )
    print(f"✓ Point cloud sphere created at {position}")


def create_line_marker(client, position, size=2.0, color=[1.0, 1.0, 0.0, 1.0]):
    """
    Create an X marker made of lines.
    Simple and always works!
    """
    x, y, z = position
    
    # Create an 'X' shape
    lines = [
        airsim.Vector3r(x - size, y - size, z),
        airsim.Vector3r(x + size, y + size, z),
        airsim.Vector3r(x + size, y - size, z),
        airsim.Vector3r(x - size, y + size, z),
        # Vertical line
        airsim.Vector3r(x, y, z - size),
        airsim.Vector3r(x, y, z + size)
    ]
    
    client.simPlotLineList(
        lines,
        color_rgba=color,
        thickness=0.3,
        duration=60.0,
        is_persistent=True
    )
    print(f"✓ Line marker created at {position}")


def create_ring_marker(client, position, radius=2.0, color=[0.0, 1.0, 1.0, 1.0]):
    """
    Create a ring (circle) made of line segments.
    Looks like a target!
    """
    num_segments = 24
    points = []
    
    for i in range(num_segments + 1):
        angle = (i / num_segments) * 2 * np.pi
        x = position[0] + radius * np.cos(angle)
        y = position[1] + radius * np.sin(angle)
        z = position[2]
        points.append(airsim.Vector3r(x, y, z))
    
    client.simPlotLineStrip(
        points,
        color_rgba=color,
        thickness=0.2,
        duration=60.0,
        is_persistent=True
    )
    print(f"✓ Ring marker created at {position}")


def demo_all_methods():
    """
    Complete demonstration of object spawning methods.
    Watch your AirSim window to see the results!
    """
    
    print("=" * 70)
    print("AirSim Object Spawning Demo")
    print("=" * 70)
    print("\nConnecting to AirSim...")
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("✓ Connected!")
    
    # Take off
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    print("✓ Drone in the air!")
    
    # Move to starting position
    print("\nMoving to starting position...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    # Clear any old markers
    print("\nClearing old markers...")
    client.simFlushPersistentMarkers()
    
    # Create 4 different targets using different methods
    print("\n" + "=" * 70)
    print("Creating visual targets...")
    print("=" * 70)
    
    print("\n1. RED SPHERE (Point Cloud Method) - North")
    create_point_cloud_sphere(
        client,
        position=[20, 0, -10],
        radius=2.0,
        color=[1.0, 0.0, 0.0, 1.0]  # Red
    )
    
    print("\n2. YELLOW X (Line Marker Method) - East")
    create_line_marker(
        client,
        position=[0, 20, -10],
        size=2.5,
        color=[1.0, 1.0, 0.0, 1.0]  # Yellow
    )
    
    print("\n3. CYAN RING (Ring Method) - South")
    create_ring_marker(
        client,
        position=[-20, 0, -10],
        radius=2.5,
        color=[0.0, 1.0, 1.0, 1.0]  # Cyan
    )
    
    print("\n4. GREEN SPHERE (Point Cloud Method) - West")
    create_point_cloud_sphere(
        client,
        position=[0, -20, -10],
        radius=2.0,
        color=[0.0, 1.0, 0.0, 1.0]  # Green
    )
    
    print("\n" + "=" * 70)
    print("All targets created! Look at your AirSim window!")
    print("=" * 70)
    
    # Fly to each target
    print("\nFlying to each target...\n")
    
    targets = [
        ([20, 0, -10], "RED SPHERE (North)"),
        ([0, 20, -10], "YELLOW X (East)"),
        ([-20, 0, -10], "CYAN RING (South)"),
        ([0, -20, -10], "GREEN SPHERE (West)")
    ]
    
    for i, (pos, name) in enumerate(targets, 1):
        print(f"{i}. Flying to {name}...")
        client.moveToPositionAsync(pos[0], pos[1], pos[2], 5).join()
        
        # Get actual position
        state = client.getMultirotorState()
        drone_pos = state.kinematics_estimated.position
        distance = np.sqrt(
            (drone_pos.x_val - pos[0])**2 +
            (drone_pos.y_val - pos[1])**2 +
            (drone_pos.z_val - pos[2])**2
        )
        
        print(f"   ✓ Reached! Distance to target: {distance:.2f}m")
        time.sleep(2)
    
    # Return to center
    print("\nReturning to center...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nThe markers will stay visible for 60 seconds.")
    print("You can continue flying around to inspect them.")
    print("\nPress Ctrl+C to stop and land.")
    
    try:
        # Keep script running so markers stay visible
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nLanding drone...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("✓ Landed safely!")


def simple_chase_demo():
    """
    Simplified version: Just spawn one target and chase it.
    This is what your actual training environment does!
    """
    
    print("=" * 70)
    print("Simple Target Chase Demo")
    print("=" * 70)
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Take off
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Move to start
    client.moveToPositionAsync(0, 0, -10, 3).join()
    
    # Spawn target
    target_pos = [15, 10, -15]
    print(f"\nSpawning target at {target_pos}...")
    
    client.simFlushPersistentMarkers()
    create_point_cloud_sphere(client, target_pos, radius=2.0)
    
    # Chase it!
    print("\nChasing target...")
    client.moveToPositionAsync(
        target_pos[0], 
        target_pos[1], 
        target_pos[2], 
        velocity=5
    ).join()
    
    # Check distance
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    distance = np.linalg.norm(
        np.array([pos.x_val, pos.y_val, pos.z_val]) - np.array(target_pos)
    )
    
    if distance < 3.0:
        print(f"\n🎯 TARGET HIT! Distance: {distance:.2f}m")
    else:
        print(f"\n❌ Missed. Distance: {distance:.2f}m")
    
    # Land
    time.sleep(2)
    print("\nLanding...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    print("\nChoose demo mode:")
    print("1. Full demo (all 4 methods)")
    print("2. Simple chase (one target)")
    
    choice = input("\nEnter 1 or 2 (or press Enter for full demo): ").strip()
    
    try:
        if choice == "2":
            simple_chase_demo()
        else:
            demo_all_methods()
    except KeyboardInterrupt:
        print("\n\n✓ Demo stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure AirSim is running before starting this script!")