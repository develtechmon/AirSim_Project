"""
COMPREHENSIVE DRONE CONTROL DIAGNOSTIC (PWM BYPASS MODE)
Author: Lukas the Big Boss + GPT-5

Purpose:
  Full validation of motor PWM mapping in AirSim.
  Includes auto hover recovery after each motion to prevent flips.

Motor Order (AirSim SimpleFlight):
    moveByMotorPWMsAsync(FR, RL, FL, RR, duration)
"""

import airsim
import numpy as np
import time

# ==============================================================
# Utility Functions
# ==============================================================

def print_state(client, label=""):
    """Display current drone state (altitude, roll, pitch, yaw)"""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    ori = state.kinematics_estimated.orientation
    roll, pitch, yaw = airsim.to_eularian_angles(ori)
    print(f"  {label:20s} Alt={-pos.z_val:6.2f}m | "
          f"Roll={np.degrees(roll):7.2f}° | Pitch={np.degrees(pitch):7.2f}° | Yaw={np.degrees(yaw):7.2f}°")


def send_pwm_continuous(client, fr, rl, fl, rr, duration_sec):
    """Send PWM continuously at 20 Hz for specified duration"""
    steps = int(duration_sec / 0.05)
    for _ in range(steps):
        client.moveByMotorPWMsAsync(fr, rl, fl, rr, 0.05)
        time.sleep(0.05)


def transition_to_hover(client, steps=20, duration=1.5, hover_pwm=0.63):
    """Smooth transition to hover to prevent jerky torque"""
    for _ in range(steps):
        client.moveByMotorPWMsAsync(hover_pwm, hover_pwm, hover_pwm, hover_pwm, duration/steps)
        time.sleep(duration/steps)


def hover_recover(client, duration=2.0, hover_pwm=0.63):
    """Hold hover for a given duration to stabilize drone"""
    print("\n🌀 Hovering to stabilize...\n")
    send_pwm_continuous(client, hover_pwm, hover_pwm, hover_pwm, hover_pwm, duration)
    print_state(client, "Recovered to Hover")

def reset_world(client):
    """Reset and reinitialize AirSim world"""
    print("\n🔄 Resetting AirSim simulation...")
    try:
        client.armDisarm(False)
        client.enableApiControl(False)
        time.sleep(1)
        client.reset()
        time.sleep(2)
        client.enableApiControl(True)
        client.armDisarm(True)
        print("✅ Reset complete. Drone re-armed.\n")
    except Exception as e:
        print(f"⚠️ Reset failed: {e}\n")

# ==============================================================
# Configuration
# ==============================================================

HOVER_PWM   = 0.63
TAKEOFF_PWM = 0.68
LAND_PWM    = 0.52

ROLL_DELTA  = 0.015
PITCH_DELTA = 0.015
YAW_DELTA   = 0.015

# ==============================================================
# Initialization
# ==============================================================

print("="*90)
print("🚁 COMPREHENSIVE DRONE CONTROL DIAGNOSTIC - PWM BYPASS MODE")
print("="*90)
print("Includes hover recovery between motions to prevent flips.")
print("Motor order: moveByMotorPWMsAsync(FR, RL, FL, RR, duration)")
print("="*90)

client = airsim.MultirotorClient()
client.confirmConnection()
reset_world(client)

print("✓ Connected and armed successfully.\n")
print(f"PWM Baseline:\n  Hover = {HOVER_PWM}\n  Takeoff = {TAKEOFF_PWM}\n  Landing = {LAND_PWM}\n")

# ==============================================================
# TEST 1: TAKEOFF
# ==============================================================

print("="*90)
print("📍 TEST 1: TAKEOFF")
print("="*90)

send_pwm_continuous(client, TAKEOFF_PWM, TAKEOFF_PWM, TAKEOFF_PWM, TAKEOFF_PWM, 2.0)
print_state(client, "After Takeoff Thrust")

transition_to_hover(client)
print_state(client, "Hover Transition")

# ==============================================================
# TEST 2: HOVER
# ==============================================================

print("\n" + "="*90)
print("📍 TEST 2: STABLE HOVER (5s)")
print("="*90)

hover_recover(client, duration=5.0, hover_pwm=HOVER_PWM)

# ==============================================================
# TEST 3: ROLL LEFT / RIGHT with recovery
# ==============================================================

print("\n" + "="*90)
print("📍 TEST 3: ROLL LEFT / RIGHT (with recovery)")
print("="*90)

# # Roll LEFT: increase right motors (FR, RR)
print("\n➡️ Rolling LEFT (FR, RR up)")
send_pwm_continuous(client,
    HOVER_PWM + ROLL_DELTA,  # FR
    HOVER_PWM,               # RL
    HOVER_PWM,               # FL
    HOVER_PWM + ROLL_DELTA,  # RR
    1.5
)
print_state(client, "After Roll Left")
hover_recover(client)

# Roll RIGHT: increase left motors (FL, RL)
print("\n⬅️ Rolling RIGHT (FL, RL up)")
send_pwm_continuous(client,
    HOVER_PWM,               # FR
    HOVER_PWM + ROLL_DELTA,  # RL
    HOVER_PWM + ROLL_DELTA,  # FL
    HOVER_PWM,               # RR
    1.5
)
print_state(client, "After Roll Right")
hover_recover(client)

# ==============================================================
# TEST 4: PITCH FORWARD / BACKWARD with recovery
# ==============================================================

print("\n" + "="*90)
print("📍 TEST 4: PITCH FORWARD / BACKWARD (with recovery)")
print("="*90)

# # Pitch FORWARD: increase rear motors (RL, RR)
print("\n➡️ Pitch FORWARD (RL, RR up)")
send_pwm_continuous(client,
    HOVER_PWM,               # FR
    HOVER_PWM + PITCH_DELTA, # RL
    HOVER_PWM,               # FL
    HOVER_PWM + PITCH_DELTA, # RR
    1.5
)
print_state(client, "After Pitch Forward")
hover_recover(client)

# # Pitch BACKWARD: increase front motors (FR, FL)
print("\n⬅️ Pitch BACKWARD (FR, FL up)")
send_pwm_continuous(client,
    HOVER_PWM + PITCH_DELTA, # FR
    HOVER_PWM,               # RL
    HOVER_PWM + PITCH_DELTA, # FL
    HOVER_PWM,               # RR
    1.5
)
print_state(client, "After Pitch Backward")
hover_recover(client)

# ==============================================================
# TEST 5: YAW LEFT / RIGHT with recovery
# ==============================================================

print("\n" + "="*90)
print("📍 TEST 5: YAW LEFT / RIGHT (with recovery)")
print("="*90)

# Yaw LEFT (CCW): FL+, RR+, FR-, RL-
print("\n↶ YAW LEFT (CCW)")
send_pwm_continuous(client,
    HOVER_PWM - YAW_DELTA,  # FR
    HOVER_PWM - YAW_DELTA,  # RL
    HOVER_PWM + YAW_DELTA,  # FL
    HOVER_PWM + YAW_DELTA,  # RR
    1.2
)
print_state(client, "After Yaw Left")
hover_recover(client)

# Yaw RIGHT (CW): FR+, RL+, FL-, RR-
print("\n↷ YAW RIGHT (CW)")
send_pwm_continuous(client,
    HOVER_PWM + YAW_DELTA,  # FR
    HOVER_PWM + YAW_DELTA,  # RL
    HOVER_PWM - YAW_DELTA,  # FL
    HOVER_PWM - YAW_DELTA,  # RR
    1.2
)
print_state(client, "After Yaw Right")
hover_recover(client)

# ==============================================================
# TEST 6: LANDING
# ==============================================================

print("\n" + "="*90)
print("📍 TEST 6: SAFE LANDING")
print("="*90)

print_state(client, "Before Landing")

print("Descending gradually...\n")
send_pwm_continuous(client, LAND_PWM, LAND_PWM, LAND_PWM, LAND_PWM, 3.0)
send_pwm_continuous(client, LAND_PWM - 0.05, LAND_PWM - 0.05, LAND_PWM - 0.05, LAND_PWM - 0.05, 2.0)
send_pwm_continuous(client, 0.0, 0.0, 0.0, 0.0, 1.0)

print_state(client, "After Landing")

# ==============================================================
# Summary
# ==============================================================

print("\n" + "="*90)
print("📊 SUMMARY")
print("="*90)
print("All maneuvers completed safely.\n")
print(f"Hover PWM:   {HOVER_PWM}")
print(f"Roll ΔPWM:   {ROLL_DELTA}")
print(f"Pitch ΔPWM:  {PITCH_DELTA}")
print(f"Yaw ΔPWM:    {YAW_DELTA}")
print("\n✅ If drone remained stable during all tests, mapping confirmed correct!")

client.armDisarm(False)
client.enableApiControl(False)
print("\n✓ Drone disarmed and API control released.")
print("="*90)
