#!/usr/bin/env python3
"""
Keyboard Controlled Drone with Optical Flow
============================================

Control your drone with keyboard for indoor flight using optical flow.

See README.md for ArduPilot parameter configuration.

Author: Created for PhD research on drone behavioral cloning
Date: November 2025
"""

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time
import threading
import math

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Install pynput for real-time control: pip install pynput")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONNECTION_STRING = 'udp:127.0.0.1:14550'
# CONNECTION_STRING = '/dev/ttyACM0'       # USB
# CONNECTION_STRING = '/dev/ttyAMA0'       # Raspberry Pi UART

TAKEOFF_ALTITUDE = 5.0      # meters
DEFAULT_SPEED = 0.5         # m/s
YAW_RATE = 30               # deg/s
VERTICAL_SPEED = 0.3        # m/s

# EKF Origin for indoor flight
ORIGIN_LAT = 47.641468
ORIGIN_LON = -122.140165
ORIGIN_ALT = 0

# ============================================================================
# DRONE STATE
# ============================================================================

class DroneState:
    def __init__(self):
        self.vehicle = None
        self.is_flying = False
        self.is_hovering = False
        self.running = True
        self.keys_pressed = set()
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.yaw_rate = 0

state = DroneState()

# ============================================================================
# DRONE FUNCTIONS
# ============================================================================

def get_altitude():
    """Get altitude from rangefinder or barometer."""
    if state.vehicle.rangefinder and state.vehicle.rangefinder.distance is not None:
        return state.vehicle.rangefinder.distance
    return state.vehicle.location.global_relative_frame.alt or 0


def connect_drone():
    print(f"\nConnecting to {CONNECTION_STRING}...")
    try:
        state.vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=30)
        print(f"Connected! Mode: {state.vehicle.mode.name}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def set_ekf_origin():
    msg = state.vehicle.message_factory.set_gps_global_origin_encode(
        0, int(ORIGIN_LAT * 1e7), int(ORIGIN_LON * 1e7), int(ORIGIN_ALT * 1000)
    )
    state.vehicle.send_mavlink(msg)
    time.sleep(0.5)


def takeoff():
    if state.is_flying:
        print("Already flying!")
        return
    
    print("\n[TAKEOFF]")
    set_ekf_origin()
    
    state.vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    
    if state.vehicle.mode.name != "GUIDED":
        print(f"Could not enter GUIDED mode: {state.vehicle.mode.name}")
        return
    
    state.vehicle.armed = True
    while not state.vehicle.armed:
        time.sleep(0.3)
    
    print("Armed! Taking off...")
    state.vehicle.simple_takeoff(TAKEOFF_ALTITUDE)
    
    while True:
        alt = get_altitude()
        if alt >= TAKEOFF_ALTITUDE * 0.8:
            break
        time.sleep(0.3)
    
    print(f"Airborne at {alt:.1f}m")
    state.is_flying = True
    state.is_hovering = True


def land():
    print("\n[LAND]")
    state.vx = state.vy = state.vz = state.yaw_rate = 0
    state.is_hovering = False
    state.vehicle.mode = VehicleMode("LAND")
    state.is_flying = False


def switch_to_guided():
    print("\n[GUIDED]")
    state.vehicle.mode = VehicleMode("GUIDED")


def hover():
    """Enter hover mode - hold position."""
    print("\n[HOVER]")
    state.vx = state.vy = state.vz = state.yaw_rate = 0
    state.is_hovering = True


def emergency_stop():
    print("\n[EMERGENCY STOP]")
    state.vx = state.vy = state.vz = state.yaw_rate = 0
    state.is_hovering = False
    state.vehicle.mode = VehicleMode("LAND")
    state.is_flying = False


def send_velocity():
    """Send velocity command to drone."""
    if not state.is_flying:
        return
    
    msg = state.vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000011111000111,
        0, 0, 0,
        state.vx, state.vy, state.vz,
        0, 0, 0,
        0, math.radians(state.yaw_rate)
    )
    state.vehicle.send_mavlink(msg)


# ============================================================================
# KEYBOARD CONTROL
# ============================================================================

def update_velocity():
    """Update velocity based on pressed keys."""
    if state.is_hovering:
        state.vx = state.vy = state.vz = state.yaw_rate = 0
        return
    
    vx, vy, vz, yaw = 0, 0, 0, 0
    
    if 'w' in state.keys_pressed: vx = DEFAULT_SPEED
    if 's' in state.keys_pressed: vx = -DEFAULT_SPEED
    if 'a' in state.keys_pressed: vy = -DEFAULT_SPEED
    if 'd' in state.keys_pressed: vy = DEFAULT_SPEED
    if 'r' in state.keys_pressed: vz = -VERTICAL_SPEED
    if 'f' in state.keys_pressed: vz = VERTICAL_SPEED
    if 'q' in state.keys_pressed: yaw = -YAW_RATE
    if 'e' in state.keys_pressed: yaw = YAW_RATE
    
    state.vx, state.vy, state.vz, state.yaw_rate = vx, vy, vz, yaw


def on_key_press(key):
    try:
        k = key.char.lower()
        state.keys_pressed.add(k)
        
        if k == 't': 
            threading.Thread(target=takeoff, daemon=True).start()
        elif k == 'l': 
            threading.Thread(target=land, daemon=True).start()
        elif k == 'g': 
            switch_to_guided()
        elif k == 'h': 
            hover()
        elif k in ['w', 's', 'a', 'd', 'q', 'e', 'r', 'f']:
            if state.is_hovering:
                print("\n[MANUAL]")
                state.is_hovering = False
                
    except AttributeError:
        if key == keyboard.Key.space: 
            emergency_stop()
        elif key == keyboard.Key.esc: 
            state.running = False


def on_key_release(key):
    try:
        state.keys_pressed.discard(key.char.lower())
    except AttributeError:
        pass


def control_loop():
    while state.running:
        if state.is_flying:
            update_velocity()
            send_velocity()
        time.sleep(0.1)


def status_loop():
    while state.running:
        if state.vehicle:
            alt = get_altitude()
            mode = state.vehicle.mode.name
            armed = "ARM" if state.vehicle.armed else "DISARM"
            ctrl = "HOVER" if state.is_hovering else "MANUAL"
            print(f"\r[{mode}][{armed}][{ctrl}] Alt:{alt:.1f}m Vel:({state.vx:.1f},{state.vy:.1f},{state.vz:.1f}) Yaw:{state.yaw_rate}   ", end='')
        time.sleep(0.2)


# ============================================================================
# SIMPLE INPUT MODE (fallback)
# ============================================================================

def simple_mode():
    print("\nSimple command mode (no real-time keyboard)")
    print("Commands: t=takeoff, l=land, g=guided, h=hover, w/s/a/d/q/e/r/f [duration], quit")
    
    while state.running:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd: continue
            
            action = cmd[0]
            duration = float(cmd[1]) if len(cmd) > 1 else 1.0
            
            if action == 't': 
                takeoff()
            elif action == 'l': 
                land()
            elif action == 'g': 
                switch_to_guided()
            elif action == 'h':
                hover()
                end_time = time.time() + duration
                while time.time() < end_time:
                    send_velocity()
                    time.sleep(0.1)
            elif action == 'quit': 
                state.running = False
            elif action in ['w', 's', 'a', 'd', 'q', 'e', 'r', 'f']:
                state.is_hovering = False
                vel_map = {
                    'w': (DEFAULT_SPEED, 0, 0, 0),
                    's': (-DEFAULT_SPEED, 0, 0, 0),
                    'a': (0, -DEFAULT_SPEED, 0, 0),
                    'd': (0, DEFAULT_SPEED, 0, 0),
                    'q': (0, 0, 0, -YAW_RATE),
                    'e': (0, 0, 0, YAW_RATE),
                    'r': (0, 0, -VERTICAL_SPEED, 0),
                    'f': (0, 0, VERTICAL_SPEED, 0),
                }
                state.vx, state.vy, state.vz, state.yaw_rate = vel_map[action]
                end_time = time.time() + duration
                while time.time() < end_time:
                    send_velocity()
                    time.sleep(0.1)
                hover()
        except KeyboardInterrupt:
            state.running = False
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def print_controls():
    print("""
╔══════════════════════════════════════════════════╗
║           KEYBOARD DRONE CONTROL                 ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║        W            Forward                      ║
║      A S D          Left / Back / Right          ║
║                                                  ║
║      Q   E          Yaw Left / Yaw Right         ║
║      R   F          Up / Down                    ║
║                                                  ║
║        T            Takeoff                      ║
║        L            Land                         ║
║        G            GUIDED mode                  ║
║        H            Hover (hold position)        ║
║                                                  ║
║      SPACE          Emergency Stop               ║
║       ESC           Quit                         ║
║                                                  ║
║  Press H to hover, WASD to move (exits hover)   ║
║                                                  ║
╚══════════════════════════════════════════════════╝
    """)


def main():
    print("\n=== KEYBOARD DRONE CONTROL ===")
    print("MTF-02 Optical Flow Indoor Flight\n")
    
    if not connect_drone():
        return
    
    print_controls()
    
    if KEYBOARD_AVAILABLE:
        print("Real-time keyboard mode. Press ESC to quit.\n")
        
        threading.Thread(target=control_loop, daemon=True).start()
        threading.Thread(target=status_loop, daemon=True).start()
        
        with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
            listener.join()
    else:
        simple_mode()
    
    print("\n\nShutting down...")
    if state.is_flying:
        land()
    if state.vehicle:
        state.vehicle.close()
    print("Done!")


if __name__ == "__main__":
    main()