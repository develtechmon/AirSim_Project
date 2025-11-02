"""
Visual Demo: Point Cloud Sphere Target for Drone Chase

Run this BEFORE training to verify that:
1. AirSim is working correctly
2. The point cloud spheres are visible
3. The drone can move and detect the targets

This is like a "pre-flight check" before actual training!
"""

import airsim
import numpy as np
import time


def create_point_cloud_sphere(client, position, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
    """
    Create a sphere made of points - this is what your training will use!
    
    Analogy: Imagine taking 150 tiny LEDs and arranging them perfectly
    to form a glowing sphere in 3D space. That's what this does!
    """
    points = []
    num_points = 150
    
    # Fibonacci sphere algorithm - ensures even distribution
    # (prevents clumping of points at poles like a bad orange peel)
    for i in range(num_points):
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
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
        duration=120.0,  # Visible for 2 minutes
        is_persistent=True
    )


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance - same as training environment uses"""
    return np.sqrt(
        (pos1[0] - pos2[0])**2 +
        (pos1[1] - pos2[1])**2 +
        (pos1[2] - pos2[2])**2
    )


def demo_sphere_chase():
    """
    Demonstrates the exact scenario your drone will train on:
    1. Spawn a red sphere at random location
    2. Chase it
    3. Check if we "hit" it (within 2m)
    4. Respawn and repeat
    """
    
    print("=" * 70)
    print("🚁 DRONE SPHERE CHASE DEMO")
    print("=" * 70)
    print("\nThis simulates what happens during training!")
    print("Watch your AirSim window to see the red sphere targets.\n")
    
    # Connect to AirSim
    print("[1/5] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("      ✓ Connected!")
    
    # Enable control
    print("[2/5] Taking control of drone...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("      ✓ Drone armed!")
    
    # Take off
    print("[3/5] Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    print("      ✓ Airborne!")
    
    # Move to starting position
    print("[4/5] Moving to starting position (0, 0, -10)...")
    client.moveToPositionAsync(0, 0, -10, 5).join()
    time.sleep(1)
    print("      ✓ Ready to start!\n")
    
    # Clear old markers
    client.simFlushPersistentMarkers()
    
    print("=" * 70)
    print("[5/5] Starting chase sequence...")
    print("=" * 70)
    
    num_targets = 3
    hit_threshold = 2.0  # Same as training environment
    
    for round_num in range(1, num_targets + 1):
        print(f"\n🎯 ROUND {round_num}/{num_targets}")
        print("-" * 70)
        
        # Spawn random target (same logic as training)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(10, 30)
        
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = np.random.uniform(-30, -10)
        
        target_pos = [target_x, target_y, target_z]
        
        print(f"\n1. Spawning target at: ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
        
        # Create visual sphere
        create_point_cloud_sphere(
            client,
            position=target_pos,
            radius=2.0,
            color=[1.0, 0.0, 0.0, 1.0]  # Red
        )
        
        print("   ✓ Red sphere spawned! (Look at AirSim window)")
        
        # Get current position
        state = client.getMultirotorState()
        drone_pos = state.kinematics_estimated.position
        start_distance = calculate_distance(
            [drone_pos.x_val, drone_pos.y_val, drone_pos.z_val],
            target_pos
        )
        
        print(f"\n2. Initial distance: {start_distance:.2f}m")
        print(f"   Chasing target...")
        
        # Chase the target
        client.moveToPositionAsync(
            target_x, target_y, target_z,
            velocity=8  # Fast chase
        ).join()
        
        # Check final distance
        state = client.getMultirotorState()
        drone_pos = state.kinematics_estimated.position
        final_distance = calculate_distance(
            [drone_pos.x_val, drone_pos.y_val, drone_pos.z_val],
            target_pos
        )
        
        print(f"\n3. Final distance: {final_distance:.2f}m")
        
        # Check if hit
        if final_distance <= hit_threshold:
            print(f"   🎉 TARGET HIT! (within {hit_threshold}m threshold)")
            print(f"   +100 reward points!")
        else:
            print(f"   ❌ Miss! (need to be within {hit_threshold}m)")
            print(f"   Distance penalty: -{final_distance * 0.01:.2f}")
        
        # Calculate altitude difference
        altitude_diff = abs(drone_pos.z_val - target_z)
        print(f"\n4. Altitude difference: {altitude_diff:.2f}m")
        if altitude_diff < 5.0:
            print(f"   ✓ Good altitude matching!")
        else:
            print(f"   ⚠️  Altitude difference too large (penalty: -{altitude_diff * 0.5:.2f})")
        
        time.sleep(2)
        
        # Clear this sphere before next round
        if round_num < num_targets:
            print(f"\n   Clearing sphere for next round...")
            client.simFlushPersistentMarkers()
            time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nYou just saw exactly what the training loop does:")
    print(f"  1. Spawn red sphere at random location")
    print(f"  2. Drone chases it")
    print(f"  3. Check if hit (within {hit_threshold}m)")
    print(f"  4. Calculate rewards/penalties")
    print(f"  5. Respawn and repeat")
    print(f"\nDuring training, the drone will learn through 300,000+ attempts!")
    
    # Return home
    print(f"\nReturning to home position...")
    client.moveToPositionAsync(0, 0, -10, 5).join()
    time.sleep(1)
    
    # Land
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ Landed safely!")
    print("\n💡 TIP: If the spheres looked good, you're ready to start training!")


def demo_multiple_spheres():
    """
    Show multiple spheres at once with different colors.
    This helps visualize the 3D space and spawning positions.
    """
    
    print("=" * 70)
    print("🎨 MULTIPLE SPHERES DEMO")
    print("=" * 70)
    print("\nShowing different spawn positions your drone might encounter.\n")
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Take off
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    client.moveToPositionAsync(0, 0, -10, 3).join()
    
    # Clear old markers
    client.simFlushPersistentMarkers()
    
    print("\nSpawning spheres at different distances and altitudes...")
    print("(Each color represents a different spawn zone)\n")
    
    # Spawn spheres at various positions
    spheres = [
        # Close targets
        ([15, 0, -10], [1.0, 0.0, 0.0, 1.0], "RED - Close (15m)"),
        ([0, 15, -15], [1.0, 0.5, 0.0, 1.0], "ORANGE - Close (15m, lower altitude)"),
        
        # Medium distance
        ([25, 10, -20], [1.0, 1.0, 0.0, 1.0], "YELLOW - Medium (27m)"),
        ([-20, 15, -12], [0.0, 1.0, 0.0, 1.0], "GREEN - Medium (25m)"),
        
        # Far targets
        ([30, 0, -25], [0.0, 0.0, 1.0, 1.0], "BLUE - Far (30m, deep)"),
        ([0, -30, -30], [0.5, 0.0, 1.0, 1.0], "PURPLE - Far (30m, very deep)")
    ]
    
    for pos, color, desc in spheres:
        create_point_cloud_sphere(client, pos, radius=2.0, color=color)
        print(f"  ✓ {desc} at ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
        time.sleep(0.5)
    
    print("\n✓ All spheres spawned!")
    print("\nLook at your AirSim window - you should see 6 colored spheres.")
    print("This shows the variety of positions your drone will learn to chase.\n")
    
    # Fly around to show the spheres
    print("Flying around to showcase the spheres...")
    print("(Press Ctrl+C to stop early)\n")
    
    viewpoints = [
        ([15, 0, -10], "Close to RED sphere"),
        ([25, 10, -20], "Near YELLOW sphere"),
        ([0, 0, -15], "Center view"),
        ([-20, 15, -12], "Near GREEN sphere"),
    ]
    
    try:
        for pos, desc in viewpoints:
            print(f"→ Moving to: {desc}")
            client.moveToPositionAsync(pos[0], pos[1], pos[2], 5).join()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user!")
    
    # Return and land
    print("\nReturning home...")
    client.moveToPositionAsync(0, 0, -10, 3).join()
    time.sleep(1)
    
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n✓ Demo complete!")
    print("\n💡 The spheres will stay visible for 2 minutes.")
    print("   You can manually fly around in AirSim to inspect them!")


def quick_test():
    """
    Ultra-simple test: Spawn one sphere and check if it's visible.
    Use this for quick verification.
    """
    
    print("=" * 70)
    print("⚡ QUICK SPHERE TEST")
    print("=" * 70)
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected to AirSim")
    
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    time.sleep(2)
    
    print("\nSpawning ONE red sphere at (20, 0, -15)...")
    client.simFlushPersistentMarkers()
    create_point_cloud_sphere(
        client,
        position=[20, 0, -15],
        radius=2.0,
        color=[1.0, 0.0, 0.0, 1.0]
    )
    
    print("\n✓ Sphere spawned!")
    print("\n👀 Look at your AirSim window.")
    print("   Do you see a RED SPHERE at position (20, 0, -15)?")
    print("\n   [YES] → You're ready to train!")
    print("   [NO]  → Check AirSim camera angle or restart AirSim")
    
    input("\nPress Enter when done...")
    
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("\n✓ Test complete!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("POINT CLOUD SPHERE DEMO FOR DRONE CHASE TRAINING")
    print("=" * 70)
    print("\nChoose a demo:")
    print("\n1. Chase Demo (Recommended) - Simulates actual training")
    print("2. Multiple Spheres - Shows various spawn positions")
    print("3. Quick Test - Just spawn one sphere to verify")
    print("=" * 70)
    
    choice = input("\nEnter 1, 2, or 3 (or press Enter for Chase Demo): ").strip()
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT: Make sure AirSim is running BEFORE continuing!")
    print("=" * 70)
    
    ready = input("\nIs AirSim running? (yes/no): ").strip().lower()
    
    if ready not in ['yes', 'y']:
        print("\n✋ Start AirSim first, then run this script again!")
        exit()
    
    print("\n🚀 Starting demo...\n")
    
    try:
        if choice == "2":
            demo_multiple_spheres()
        elif choice == "3":
            quick_test()
        else:
            demo_sphere_chase()
    except KeyboardInterrupt:
        print("\n\n✓ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is AirSim running?")
        print("  2. Is the drone in a flyable environment?")
        print("  3. Try restarting AirSim and this script")