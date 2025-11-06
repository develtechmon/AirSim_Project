"""
AirSim Drone - Direct Motor PWM Control (Bypass Mode, Corrected Mapping + Reset)
Author: Lukas the Big Boss + GPT-5

Features:
- Full manual PWM control of each motor
- Corrected motor mapping (FR, RL, FL, RR)
- Keyboard controls for pitch, roll, yaw, throttle
- On-screen HUD showing current PWM values
- Instant simulation reset using 'T' key

Mapping:
  W/S : Pitch Forward/Backward
  A/D : Roll Left/Right
  Q/E : Yaw Left/Right
  R/F : Throttle Up/Down
  SPACE : Hover Reset
  T : Reset Simulation
  ESC : Quit
"""

import airsim
import pygame
import numpy as np
import time

# -----------------------------------
# AirSim Setup
# -----------------------------------
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("✅ Connected to AirSim drone (PWM Mode)")

# -----------------------------------
# Pygame Setup
# -----------------------------------
pygame.init()
win = pygame.display.set_mode((420, 200))
pygame.display.set_caption("AirSim Direct PWM Control (Bypass Mode)")

font = pygame.font.SysFont("Consolas", 16)
clock = pygame.time.Clock()
running = True

# -----------------------------------
# PWM Parameters
# -----------------------------------
hover_pwm = 0.60          # baseline hover (tune as needed)
pwm_step = 0.03           # PWM delta per keypress
duration = 0.05           # duration per command
motor_pwm = np.array([hover_pwm] * 4, dtype=float)

# AirSim motor order
FR, RL, FL, RR = 0, 1, 2, 3  # fixed order for moveByMotorPWMsAsync()

YAW_DIR = 1.0  # reverse yaw if needed by setting -1.0


# -----------------------------------
# Reset Function
# -----------------------------------
def reset_simulation():
    """
    Fully resets AirSim drone:
      - Disarm
      - Reset world physics
      - Re-enable API control
      - Arm drone again
      - Perform takeoff to hover altitude
    """
    print("\n🔄 Resetting AirSim simulation...")

    try:
        client.armDisarm(False)
        client.enableApiControl(False)
        time.sleep(1)

        # Reset the environment
        client.reset()
        time.sleep(2)

        # Reconnect and re-arm
        client.enableApiControl(True)
        client.armDisarm(True)
        time.sleep(1)

        # Simple re-takeoff using PWM
        print("🚁 Taking off again...")
        client.moveByMotorPWMsAsync(0.65, 0.65, 0.65, 0.65, 2).join()
        print("✅ Reset complete, hovering in bypass mode.\n")

    except Exception as e:
        print(f"⚠️ Reset failed: {e}")


# -----------------------------------
# Initial Takeoff
# -----------------------------------
client.moveByMotorPWMsAsync(0.65, 0.65, 0.65, 0.65, 2).join()
print("🚁 Drone lifted - entering manual PWM control\n")


# -----------------------------------
# Main Control Loop
# -----------------------------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    motor_pwm[:] = hover_pwm  # reset each frame to neutral hover

    # ----- PITCH (W/S) -----
    if keys[pygame.K_w]:  # forward (+Y)
        motor_pwm[RL] += pwm_step  # rear
        motor_pwm[RR] += pwm_step
    if keys[pygame.K_s]:  # backward (-Y)
        motor_pwm[FL] += pwm_step  # front
        motor_pwm[FR] += pwm_step

    # ----- ROLL (A/D) -----
    if keys[pygame.K_a]:  # left (-X)
        motor_pwm[FR] += pwm_step
        motor_pwm[RR] += pwm_step
    if keys[pygame.K_d]:  # right (+X)
        motor_pwm[FL] += pwm_step
        motor_pwm[RL] += pwm_step

    # ----- YAW (Q/E) -----
    if keys[pygame.K_q]:  # yaw left (CCW)
        motor_pwm[FL] += pwm_step * YAW_DIR
        motor_pwm[RR] += pwm_step * YAW_DIR
        motor_pwm[FR] -= pwm_step * YAW_DIR
        motor_pwm[RL] -= pwm_step * YAW_DIR
    if keys[pygame.K_e]:  # yaw right (CW)
        motor_pwm[FR] += pwm_step * YAW_DIR
        motor_pwm[RL] += pwm_step * YAW_DIR
        motor_pwm[FL] -= pwm_step * YAW_DIR
        motor_pwm[RR] -= pwm_step * YAW_DIR

    # ----- THROTTLE (R/F) -----
    if keys[pygame.K_r]:
        motor_pwm += pwm_step
    if keys[pygame.K_f]:
        motor_pwm -= pwm_step

    # Hover reset
    if keys[pygame.K_SPACE]:
        motor_pwm[:] = hover_pwm

    # ----- Simulation Reset -----
    if keys[pygame.K_t]:
        reset_simulation()
        motor_pwm[:] = hover_pwm
        time.sleep(1)  # prevent immediate re-trigger

    # Quit
    if keys[pygame.K_ESCAPE]:
        running = False
        break

    # Clamp PWM range and send command
    motor_pwm = np.clip(motor_pwm, 0.0, 1.0)
    client.moveByMotorPWMsAsync(
        float(motor_pwm[FR]),
        float(motor_pwm[RL]),
        float(motor_pwm[FL]),
        float(motor_pwm[RR]),
        duration
    )

    # --- On-screen HUD ---
    win.fill((25, 25, 25))
    hud_text = font.render(
        f"FR:{motor_pwm[FR]:.2f}  RL:{motor_pwm[RL]:.2f}  FL:{motor_pwm[FL]:.2f}  RR:{motor_pwm[RR]:.2f}",
        True, (200, 255, 200))
    win.blit(hud_text, (20, 80))
    info_text = font.render("[T] Reset Simulation  [SPACE] Hover  [ESC] Quit", True, (180, 180, 255))
    win.blit(info_text, (20, 120))
    pygame.display.flip()

    clock.tick(20)  # 20 Hz control loop


# -----------------------------------
# Cleanup
# -----------------------------------
print("\n🛑 Landing and shutdown...")
client.moveByMotorPWMsAsync(0.45, 0.45, 0.45, 0.45, 2).join()
client.armDisarm(False)
client.enableApiControl(False)
pygame.quit()
print("✅ Session Ended.")
