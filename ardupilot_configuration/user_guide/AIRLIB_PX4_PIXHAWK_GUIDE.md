# Complete Step-by-Step Guide: AirLib + PX4 + Pixhawk

## Overview: What We're Building

```
YOUR DEVELOPMENT PIPELINE:

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SIMULATION (On your Windows PC)                                    │
│                                                                             │
│   AirSim (Windows) ←──UDP──→ PX4 SITL (WSL2)                               │
│        ↑                                                                    │
│        │                                                                    │
│   Python API (same code!)                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          Train your PPO model
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: REAL DRONE                                                         │
│                                                                             │
│   Raspberry Pi 4B ──Serial/USB──→ Pixhawk (PX4 firmware)                   │
│        ↑                              ↓                                     │
│   DroneServer + Your Code         Real Motors                               │
│   (same Python API!)                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Setting Up PX4 SITL + AirSim (Simulation)

This replaces your current ArduPilot SITL setup for the PX4 pathway.

### Step 1.1: Install PX4 SITL in WSL2

Open your WSL2 Ubuntu terminal:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install git
sudo apt-get install git -y

# Clone PX4 Autopilot (this takes a while - lots of submodules)
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive

# Run the PX4 setup script (installs all dependencies)
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-sim-tools

# Restart terminal or source bashrc
source ~/.bashrc

# Navigate to PX4 directory
cd ~/PX4-Autopilot
```

### Step 1.2: Build PX4 SITL

```bash
cd ~/PX4-Autopilot

# Build the SITL version (Software-In-The-Loop)
# This takes about 2-5 minutes first time
make px4_sitl_default none_iris

# If successful, you'll see:
#   ______  __   __    ___ 
#   | ___ \ \ \ / /   /   |
#   | |_/ /  \ V /   / /| |
#   | __/   / \ \  / /_| |
#   | |    / /^\ \ \___  |
#   \_|    \/   \/     |_/
#   px4 starting.
```

Press `Ctrl+C` to stop it for now.

### Step 1.3: Configure AirSim for PX4

On your **Windows** machine, edit or create `Documents\AirSim\settings.json`:

```json
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ClockType": "SteppableClock",
    "Vehicles": {
        "PX4": {
            "VehicleType": "PX4Multirotor",
            "UseSerial": false,
            "LockStep": true,
            "UseTcp": true,
            "TcpPort": 4560,
            "ControlIp": "remote",
            "ControlPortLocal": 14540,
            "ControlPortRemote": 14580,
            "LocalHostIp": "YOUR_WSL_ADAPTER_IP",
            "Sensors": {
                "Barometer": {
                    "SensorType": 1,
                    "Enabled": true,
                    "PressureFactorSigma": 0.0001825
                }
            },
            "Parameters": {
                "NAV_RCL_ACT": 0,
                "NAV_DLL_ACT": 0,
                "COM_OBL_ACT": 1,
                "LPE_LAT": 47.641468,
                "LPE_LON": -122.140165
            }
        }
    }
}
```

### Step 1.4: Find Your WSL2 IP Addresses

On **Windows PowerShell**:
```powershell
# Get the WSL adapter IP (Windows side)
ipconfig | findstr "WSL"
# Look for something like: 172.23.128.1
```

In **WSL2 Ubuntu**:
```bash
# Get WSL2's IP address
ip addr show eth0 | grep inet
# Look for something like: 172.23.143.4
```

Update your `settings.json`:
- Replace `"LocalHostIp": "YOUR_WSL_ADAPTER_IP"` with your Windows WSL adapter IP (e.g., `"172.23.128.1"`)

### Step 1.5: Configure PX4 for Remote Connection

In WSL2, before running PX4:
```bash
# Set environment variable for remote AirSim connection
export PX4_SIM_HOST_ADDR=172.23.128.1  # Your Windows WSL adapter IP
```

### Step 1.6: Open Firewall Ports (Windows)

Open PowerShell as Administrator:
```powershell
# Allow incoming TCP on port 4560 (PX4 simulator connection)
netsh advfirewall firewall add rule name="PX4 SITL TCP" dir=in action=allow protocol=TCP localport=4560

# Allow incoming UDP on port 14540 (MAVLink)
netsh advfirewall firewall add rule name="PX4 MAVLink UDP" dir=in action=allow protocol=UDP localport=14540
```

### Step 1.7: Run PX4 SITL + AirSim

**Terminal 1 (WSL2)** - Start PX4 SITL:
```bash
cd ~/PX4-Autopilot

# Set remote host
export PX4_SIM_HOST_ADDR=172.23.128.1

# Run PX4 SITL for AirSim
make px4_sitl_default none_iris
```

Wait for:
```
INFO [simulator] Waiting for simulator to connect on TCP port 4560
```

**On Windows** - Start AirSim:
- Launch your AirSim environment (Blocks or custom)
- The drone should appear and connect

You should see in PX4 console:
```
INFO [simulator] Simulator connected on TCP port 4560
INFO [mavlink] partner IP: 172.23.128.1
INFO [ecl/EKF] EKF GPS checks passed
```

### Step 1.8: Test with Python API

Create a test script on Windows:

```python
# test_px4_airsim.py
import airsim
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")

# Enable API control
client.enableApiControl(True)
print("API control enabled")

# Arm the drone
client.armDisarm(True)
print("Armed!")

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

# Move using the SAME API you'll use on real drone
print("Moving forward...")
client.moveByVelocityAsync(2, 0, 0, 5).join()

# Hover
print("Hovering...")
client.hoverAsync().join()
time.sleep(3)

# Land
print("Landing...")
client.landAsync().join()

# Disarm
client.armDisarm(False)
client.enableApiControl(False)

print("Done!")
```

Run it:
```bash
python test_px4_airsim.py
```

---

## Part 2: Flash PX4 Firmware to Pixhawk

### Step 2.1: Download QGroundControl

1. Download from: https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html
2. Install on Windows

### Step 2.2: Connect Pixhawk

1. Connect Pixhawk to your PC via USB
2. Open QGroundControl
3. It should detect the Pixhawk automatically

### Step 2.3: Flash PX4 Firmware

1. In QGroundControl, go to **Vehicle Setup** (gear icon) → **Firmware**
2. Disconnect and reconnect Pixhawk when prompted
3. Select **PX4 Pro Stable Release**
4. Choose your vehicle frame (e.g., Quadrotor X)
5. Wait for firmware to flash (takes 1-2 minutes)

### Step 2.4: Calibrate Sensors

In QGroundControl:
1. **Compass** - Follow the calibration dance
2. **Gyroscope** - Keep it still
3. **Accelerometer** - 6-position calibration
4. **Radio** - Calibrate your RC transmitter (optional for API control)

### Step 2.5: Configure for Offboard Control

Set these parameters in QGroundControl (Parameters tab):

```
# Allow offboard control without RC
COM_RC_IN_MODE = 1       # Allow joystick/offboard
NAV_RCL_ACT = 0          # No RC loss failsafe (ONLY FOR TESTING!)
NAV_DLL_ACT = 0          # No data link loss failsafe (ONLY FOR TESTING!)

# Enable offboard control
COM_OBL_ACT = 1          # Return on offboard loss

# Attitude rate limits (for flips, increase these)
MC_ROLLRATE_MAX = 360    # deg/s (increase for flips)
MC_PITCHRATE_MAX = 360   # deg/s
```

⚠️ **WARNING**: Disabling failsafes is DANGEROUS for real flights! Only do this for controlled testing!

---

## Part 3: Build AirLib for Raspberry Pi

### Step 3.1: Set Up Raspberry Pi

On your Raspberry Pi 4B:

```bash
# Install Ubuntu 22.04 (recommended) or use Raspberry Pi OS 64-bit

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    python3-pip \
    python3-dev
```

### Step 3.2: Clone AirSim on Raspberry Pi

```bash
cd ~
git clone https://github.com/microsoft/AirSim.git
cd AirSim
```

### Step 3.3: Build AirLib and Tools

```bash
cd ~/AirSim

# Run setup script
./setup.sh

# Build AirSim (this takes a LONG time on RPi - ~30-60 minutes)
./build.sh
```

If building takes too long, you can **cross-compile** on a faster machine. But let's try native build first.

### Step 3.4: Build MavLinkTest and DroneServer

```bash
cd ~/AirSim

# The build.sh should have built these, but verify:
ls -la ./build_release/output/bin/

# You should see:
#   MavLinkTest
#   DroneServer
#   DroneShell
```

If not present, build manually:
```bash
cd ~/AirSim/MavLinkCom
mkdir build && cd build
cmake ..
make -j4

cd ~/AirSim/DroneServer
mkdir build && cd build
cmake ..
make -j4
```

---

## Part 4: Connect Raspberry Pi to Pixhawk

### Step 4.1: Physical Connection

```
Raspberry Pi 4B                    Pixhawk
┌─────────────┐                   ┌─────────────┐
│             │                   │             │
│    USB ─────┼───────────────────┼── USB       │
│             │     OR            │             │
│   GPIO TX ──┼───────────────────┼── TELEM2 RX │
│   GPIO RX ──┼───────────────────┼── TELEM2 TX │
│   GND ──────┼───────────────────┼── GND       │
│             │                   │             │
└─────────────┘                   └─────────────┘
```

**Option A: USB Connection (Easiest)**
- Just connect USB cable from RPi to Pixhawk
- Device appears as `/dev/ttyACM0`

**Option B: UART/Serial Connection (Better for production)**
- Connect to TELEM2 port on Pixhawk
- Device appears as `/dev/ttyAMA0` or `/dev/serial0`

### Step 4.2: Test Connection with MavLinkTest

```bash
cd ~/AirSim/build_release/output/bin

# Test USB connection
./MavLinkTest -serial:/dev/ttyACM0,115200 -logdir:.

# OR for UART
./MavLinkTest -serial:/dev/ttyAMA0,921600 -logdir:.
```

You should see heartbeat messages from PX4.

Test basic commands:
```
?                    # Show help
arm                  # Arm motors (BE CAREFUL - props may spin!)
takeoff 2            # Takeoff to 2 meters
land                 # Land
```

### Step 4.3: Run DroneServer

**Terminal 1** - Start MavLinkTest as proxy:
```bash
cd ~/AirSim/build_release/output/bin
./MavLinkTest -serial:/dev/ttyACM0,115200 -logdir:. -proxy:127.0.0.1:14560
```

**Terminal 2** - Start DroneServer:
```bash
cd ~/AirSim/build_release/output/bin
./DroneServer 0
```

### Step 4.4: Create AirSim Settings on RPi

```bash
mkdir -p ~/Documents/AirSim
nano ~/Documents/AirSim/settings.json
```

Add:
```json
{
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "Vehicles": {
        "PX4": {
            "VehicleType": "PX4Multirotor",
            "UseSerial": false
        }
    }
}
```

Note: `"UseSerial": false` because MavLinkTest is handling serial and proxying via UDP.

---

## Part 5: Deploy Your Code to Real Drone

### Step 5.1: Install Python AirSim on RPi

```bash
pip3 install airsim
pip3 install numpy
```

### Step 5.2: Create Your Flight Script

```python
#!/usr/bin/env python3
# real_drone_flight.py
# THIS IS THE SAME CODE THAT WORKS IN SIMULATION!

import airsim
import time

def main():
    # Connect to DroneServer on localhost
    # In simulation: connects to AirSim
    # On real drone: connects to DroneServer → PX4 → Real motors!
    client = airsim.MultirotorClient(ip="127.0.0.1")
    client.confirmConnection()
    print("Connected!")
    
    # Enable API control
    client.enableApiControl(True)
    print("API control enabled")
    
    # Arm
    print("Arming...")
    client.armDisarm(True)
    time.sleep(2)
    
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    print("Airborne!")
    
    # Hover for 5 seconds
    print("Hovering...")
    time.sleep(5)
    
    # Move forward 3 meters at 1 m/s
    print("Moving forward...")
    client.moveByVelocityAsync(1, 0, 0, 3).join()
    
    # Your trained neural network would go here!
    # action = your_ppo_model.predict(observation)
    # client.moveByAngleRatesThrottleAsync(
    #     action[0], action[1], action[2], action[3], 0.1
    # ).join()
    
    # Land
    print("Landing...")
    client.landAsync().join()
    
    # Disarm
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("Flight complete!")

if __name__ == "__main__":
    main()
```

### Step 5.3: Run Your Flight

```bash
# Make sure DroneServer is running first!
python3 real_drone_flight.py
```

---

## Part 6: Deploy Your Trained PPO Model

### Step 6.1: Structure for Deployment

```python
#!/usr/bin/env python3
# deploy_ppo_model.py

import airsim
import numpy as np
import time
# Import your trained model (TensorFlow, PyTorch, etc.)
# from your_model import PPOModel

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient(ip="127.0.0.1")
        self.client.confirmConnection()
        # self.model = PPOModel.load("trained_ppo_model.pth")
        
    def get_observation(self):
        """Get current state - SAME as training!"""
        state = self.client.getMultirotorState()
        
        # Position (NED frame)
        pos = state.kinematics_estimated.position
        
        # Velocity
        vel = state.kinematics_estimated.linear_velocity
        
        # Orientation (quaternion to euler)
        q = state.kinematics_estimated.orientation
        # Convert quaternion to roll, pitch, yaw...
        
        # Angular velocity
        ang_vel = state.kinematics_estimated.angular_velocity
        
        # Create observation vector (same format as training!)
        obs = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            # roll, pitch, yaw,
            ang_vel.x_val, ang_vel.y_val, ang_vel.z_val
        ])
        
        return obs
    
    def execute_action(self, action):
        """Execute action from neural network"""
        # action = [roll_rate, pitch_rate, yaw_rate, throttle]
        
        # Scale actions to appropriate ranges
        roll_rate = action[0] * 16.0    # Scale to rad/s
        pitch_rate = action[1] * 16.0
        yaw_rate = action[2] * 2.0
        throttle = (action[3] + 1) / 2  # Scale from [-1,1] to [0,1]
        
        # Execute! Same API as simulation
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate, pitch_rate, yaw_rate, throttle, 0.05
        ).join()
    
    def run(self):
        """Main control loop"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        print("Taking off...")
        self.client.takeoffAsync().join()
        
        print("Running neural network control...")
        try:
            for step in range(1000):  # Run for 1000 steps
                # Get observation
                obs = self.get_observation()
                
                # Get action from trained model
                # action = self.model.predict(obs)
                action = np.array([0, 0, 0, 0])  # Placeholder - hover
                
                # Execute action
                self.execute_action(action)
                
                time.sleep(0.05)  # 20 Hz control loop
                
        except KeyboardInterrupt:
            print("Interrupted!")
        
        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == "__main__":
    controller = DroneController()
    controller.run()
```

---

## Quick Reference: Complete Startup Sequence

### For Simulation (Testing):

```bash
# Terminal 1 (WSL2): Start PX4 SITL
cd ~/PX4-Autopilot
export PX4_SIM_HOST_ADDR=172.23.128.1
make px4_sitl_default none_iris

# Windows: Start AirSim environment

# Terminal 2 (Windows): Run your Python code
python your_script.py
```

### For Real Drone:

```bash
# On Raspberry Pi:

# Terminal 1: MavLinkTest proxy
cd ~/AirSim/build_release/output/bin
./MavLinkTest -serial:/dev/ttyACM0,115200 -logdir:. -proxy:127.0.0.1:14560

# Terminal 2: DroneServer
cd ~/AirSim/build_release/output/bin
./DroneServer 0

# Terminal 3: Your flight code
python3 your_script.py
```

---

## Troubleshooting

### Issue: PX4 SITL doesn't connect to AirSim
- Check firewall ports (4560 TCP, 14540 UDP)
- Verify IP addresses in settings.json
- Check `PX4_SIM_HOST_ADDR` environment variable

### Issue: MavLinkTest can't find Pixhawk
- Check USB connection: `ls /dev/ttyACM*`
- Try different baud rate: `115200` or `57600`
- Check permissions: `sudo chmod 666 /dev/ttyACM0`

### Issue: DroneServer won't start
- Make sure MavLinkTest is running with `-proxy:127.0.0.1:14560`
- Check settings.json has `"UseSerial": false`

### Issue: Drone doesn't arm
- Check PX4 parameters for failsafe settings
- Verify GPS lock in QGroundControl
- Check battery voltage

---

## Summary

You now have a complete pathway:

1. ✅ **Simulate** with PX4 SITL + AirSim (same setup as your ArduPilot but with PX4)
2. ✅ **Train** your PPO model using AirSim Python API
3. ✅ **Flash** PX4 firmware to Pixhawk
4. ✅ **Deploy** AirLib/DroneServer on Raspberry Pi
5. ✅ **Run** the SAME Python code on real drone

The key insight: **Your Python code doesn't change between simulation and real drone!**

```python
# This works in BOTH simulation AND real drone:
client.moveByAngleRatesThrottleAsync(roll_rate, pitch_rate, yaw_rate, throttle, duration)
```

Good luck with your PhD research! 🚁