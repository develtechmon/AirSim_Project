import airsim
import time

def simple_collision_test():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("🚁 Taking off for collision test...")
    client.takeoffAsync().join()
    
    print("💥 Flying forward into obstacle...")
    # Fly forward at high speed to cause collision
    client.moveByVelocityAsync(10, 0, 0, 5)  # 10 m/s forward for 5 seconds
    
    # Monitor for collision
    for i in range(50):  # Check for 5 seconds
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"🔥 COLLISION DETECTED!")
            print(f"   Time: {collision_info.time_stamp}")
            print(f"   Impact point: {collision_info.impact_point}")
            print(f"   Object hit: {collision_info.object_name}")
            break
        time.sleep(0.1)

if __name__ == "__main__":
    simple_collision_test()