# 🚁 Complete User Guide: AirSim + ArduPilot SITL + WSL2 on Windows

## 📖 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Get Your IP Addresses](#step-1-get-your-ip-addresses)
3. [Step 2: Configure AirSim settings.json](#step-2-configure-airsim-settingsjson)
4. [Step 3: Start ArduPilot SITL](#step-3-start-ardupilot-sitl)
5. [Step 4: Start AirSim](#step-4-start-airsim)
6. [Step 5: Test Connection](#step-5-test-connection)
7. [Step 6: Arm and Takeoff](#step-6-arm-and-takeoff)
8. [Step 7: Run Python Scripts](#step-7-run-python-scripts)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Prerequisites

✅ **Required Software:**
- Windows 10/11
- WSL2 (Ubuntu 20.04 or 22.04)
- ArduPilot installed in WSL2
- AirSim (Unreal Engine) on Windows
- Python 3.x with dronekit and airsim packages

✅ **Verify WSL2 Installation:**
```powershell
wsl --version
```

---

## Step 1: Get Your IP Addresses

You need TWO IP addresses for this setup to work.

### 🔍 1A. Get Windows WSL Adapter IP

**Open Windows PowerShell or Command Prompt:**

```cmd
ipconfig
```

**Look for this section:**
```
Ethernet adapter vEthernet (WSL (Hyper-V firewall)):
   IPv4 Address. . . . . . . . . . . : 172.23.128.1
   Subnet Mask . . . . . . . . . . . : 255.255.240.0
```

**Write down:** `172.23.128.1` (your number will be similar but might differ)

📝 **This is your "Windows IP"** - Keep this handy!

---

### 🔍 1B. Get WSL2 Ubuntu IP

**Open WSL2 Terminal (Ubuntu):**

```bash
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1
```

**Example output:**
```
172.23.143.43
```

📝 **This is your "WSL2 IP"** - Keep this handy too!

---

### 📋 Summary Template

Fill in your values:

```
Windows WSL Adapter IP: _______________ (e.g., 172.23.128.1)
WSL2 Ubuntu IP:        _______________ (e.g., 172.23.143.43)
```

⚠️ **IMPORTANT:** The WSL2 IP can change after Windows reboot! You may need to check it again after restarting your computer.

---

## Step 2: Configure AirSim settings.json

### 📂 2A. Locate settings.json

**Windows Path:**
```
C:\Users\YourUsername\Documents\AirSim\settings.json
```

If the folder doesn't exist, create it:
- Right-click in Documents folder
- New → Folder → Name it "AirSim"

---

### ✏️ 2B. Create/Edit settings.json

Open with Notepad and paste this (replace IPs with YOUR values from Step 1):

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "OriginGeopoint": {
    "Latitude": 37.334947,
    "Longitude": -122.012715,
    "Altitude": 583
  },
  "Vehicles": {
    "Copter": {
      "VehicleType": "ArduCopter",
      "UseSerial": false,
      "LocalHostIp": "172.23.128.1",
      "UdpIp": "172.23.143.43",
      "UdpPort": 9003,
      "ControlPort": 9002,
      "AutoCreate": true
    }
  }
}
```

**⚠️ CRITICAL: Replace these values:**
- `"LocalHostIp": "172.23.128.1"` → Use YOUR Windows IP from Step 1A
- `"UdpIp": "172.23.143.43"` → Use YOUR WSL2 IP from Step 1B

**Save the file** (Ctrl+S)

---

### ✅ 2C. Verify settings.json

Double-check:
- ✅ File is in `C:\Users\YourUsername\Documents\AirSim\settings.json`
- ✅ Both IP addresses match your values from Step 1
- ✅ File is valid JSON (no extra commas, all brackets closed)

---

## Step 3: Start ArduPilot SITL

### 🚀 3A. Open WSL2 Terminal

**Option 1:** Windows Terminal → Select "Ubuntu"
**Option 2:** Start Menu → Search "Ubuntu"

---

### 🎯 3B. Run sim_vehicle.py Command

**Navigate to ArduCopter directory:**
```bash
cd ~/ardupilot/ArduCopter
```

**Start SITL with your Windows IP:**
```bash
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map --sim-address=172.23.128.1
```

**⚠️ REPLACE:** `172.23.128.1` with YOUR Windows IP from Step 1A

---

### ✅ 3C. Verify SITL Started Correctly

**You should see:**
```
Starting SITL Airsim
Bind SITL sensor input at 172.23.128.1:9003
AirSim control interface set to 172.23.128.1:9002
Starting sketch 'ArduCopter'
Starting SITL input

Connect tcp:127.0.0.1:5760 source_system=255
Loaded module console
Loaded module map
Waiting for heartbeat from tcp:127.0.0.1:5760
```

**✅ GOOD SIGNS:**
- `Bind SITL sensor input at 172.23.128.1:9003` (shows YOUR Windows IP)
- `AirSim control interface set to 172.23.128.1:9002` (shows YOUR Windows IP)
- Map window opens
- Console window opens

**❌ BAD SIGNS:**
- `Bind SITL sensor input at 127.0.0.1:9003` (wrong IP!)
- Error messages about connection refused
- No map or console windows

**If you see the bad signs, go back to Step 3B and verify your IP address!**

---

### 🔄 3D. Leave SITL Running

**DO NOT CLOSE** this terminal! Keep it running.

You should see:
```
MAV> Waiting for heartbeat from tcp:127.0.0.1:5760
```

This is normal - it's waiting for AirSim to connect.

---

## Step 4: Start AirSim

### 🎮 4A. Launch AirSim

**Option 1:** If using Unreal Engine Project
- Open Unreal Engine
- Load your AirSim project
- Click "Play" button

**Option 2:** If using Pre-built Binary
- Navigate to your AirSim executable location
- Double-click `MSBuild2019.exe` or similar

---

### ⏳ 4B. Wait for Connection

**What happens:**
1. AirSim window opens
2. Environment loads (may take 10-30 seconds)
3. You may see "Waiting for connection..." message
4. After a few seconds, a drone should appear in the environment

---

### ✅ 4C. Verify Connection

**In your WSL2 Terminal (from Step 3), you should now see:**
```
MAV> Detected vehicle 1:1 on link 0
STABILIZE>
```

**✅ SUCCESS!** ArduPilot is now connected to AirSim!

**In AirSim window:**
- ✅ A quadcopter should be visible on the ground
- ✅ You can move camera around (right-click + drag)

---

## Step 5: Test Connection

### 🧪 5A. Test Basic Commands

**In the MAVProxy console (WSL2 Terminal), type:**

```bash
# Check mode
mode
```

**Expected output:**
```
STABILIZE>
```

---

### 🧪 5B. Disable Pre-Arm Checks (SITL Only!)

**In MAVProxy console:**
```bash
param set ARMING_CHECK 0
```

**You should see:**
```
ARMING_CHECK = 0
```

⚠️ **WARNING:** NEVER use this on a real drone! This disables safety checks and is only for simulation.

---

## Step 6: Arm and Takeoff

### 🚁 6A. Switch to GUIDED Mode

**In MAVProxy console:**
```bash
mode guided
```

**Wait for:**
```
GUIDED>
```

---

### 🔧 6B. Arm the Vehicle

**In MAVProxy console:**
```bash
arm throttle
```

**You should see:**
```
ARMED
```

**In AirSim:** Propellers should start spinning!

---

### 🚀 6C. Takeoff

**In MAVProxy console:**
```bash
takeoff 30
```

This commands the drone to takeoff to 30 meters altitude.

**What you'll see:**
```
Altitude: 5.2m
Altitude: 10.8m
Altitude: 15.3m
...
Altitude: 29.8m
```

**In AirSim:** The drone lifts off and climbs to 30 meters!

---

### ✅ 6D. Success!

Your drone is now:
- ✅ Flying in AirSim
- ✅ Controlled by ArduPilot SITL
- ✅ Ready for autonomous missions or DroneKit control

---


## Troubleshooting

### ❌ Problem: Error 10049 in AirSim

**Cause:** Wrong IP addresses in settings.json

**Solution:**
1. Check your IPs again (Step 1)
2. Update settings.json (Step 2)
3. Restart AirSim

---

### ❌ Problem: "Bind SITL sensor input at 127.0.0.1:9003"

**Cause:** Not using correct `--sim-address` parameter

**Solution:**
Ensure you're using:
```bash
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map --sim-address=YOUR_WINDOWS_IP
```

**Don't forget `--sim-address=YOUR_WINDOWS_IP`!**

---

### ❌ Problem: AirSim Freezes

**Cause:** ArduPilot SITL not running first, or wrong IPs

**Solution:**
1. **Always start ArduPilot SITL first** (Step 3)
2. **Wait** until you see "Waiting for heartbeat..."
3. **Then** start AirSim (Step 4)

---

### ❌ Problem: "Pre-arm: GPS not healthy"

**Cause:** GPS not initialized yet

**Solution:**
```bash
param set ARMING_CHECK 0
```

Then try arming again.

---

### ❌ Problem: WSL2 IP Changed After Reboot

**Cause:** WSL2 assigns dynamic IPs

**Solution:**
1. Get new WSL2 IP: `ip addr show eth0 | grep "inet "`
2. Update `UdpIp` in settings.json
3. Restart AirSim

---

### ❌ Problem: MAVProxy Shows "Connection refused"

**Cause:** ArduCopter binary not running

**Solution:**
Check for error messages in the SITL output. Try rebuilding:
```bash
cd ~/ardupilot
./waf configure --board sitl
./waf copter
```

---

## Quick Reference

### 📋 Command Cheat Sheet

**Get IPs:**
```cmd
# Windows
ipconfig

# WSL2
ip addr show eth0 | grep "inet "
```

**Start SITL:**
```bash
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map --sim-address=YOUR_WINDOWS_IP
```

**MAVProxy Commands:**
```bash
# Disable checks (SITL only!)
param set ARMING_CHECK 0

# Arm and takeoff
mode guided
arm throttle
takeoff 30

# Land
mode land

# Get status
status
```

**Python Connection:**
```python
# DroneKit
from dronekit import connect
vehicle = connect('tcp:127.0.0.1:5760', wait_ready=True)

# AirSim
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
```

---

### 📂 File Locations

**settings.json:**
```
C:\Users\YourUsername\Documents\AirSim\settings.json
```

**ArduPilot SITL:**
```
~/ardupilot/ArduCopter/
```

**Python Scripts:**
```
Wherever you saved them (e.g., C:\Users\YourUsername\Documents\drone_scripts\)
```

---

### 🔄 Full Startup Sequence

1. ✅ Open WSL2 Terminal
2. ✅ Run: `sim_vehicle.py -v ArduCopter -f airsim-copter --console --map --sim-address=YOUR_WINDOWS_IP`
3. ✅ Wait for "Waiting for heartbeat..."
4. ✅ Start AirSim (MSBuild2019.exe or Unreal)
5. ✅ Wait for drone to appear
6. ✅ See "STABILIZE>" in MAVProxy
7. ✅ Run Python scripts

---

### 💾 Save Your IPs

Create a text file with your IPs so you don't have to look them up every time:

**my_ips.txt:**
```
Windows WSL IP: 172.23.128.1
WSL2 IP: 172.23.143.43

Command:
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map --sim-address=172.23.128.1
```

---

## 🎉 Success Checklist

When everything is working, you should have:

- ✅ WSL2 Terminal showing "STABILIZE>" or "GUIDED>"
- ✅ AirSim window showing a quadcopter
- ✅ Can arm and takeoff using MAVProxy commands
- ✅ Python can connect via DroneKit (tcp:127.0.0.1:5760)
- ✅ Python can connect via AirSim API
- ✅ Drone responds to commands in simulation

---

## 📚 Next Steps

Now that your setup is working:

1. **Stage 1:** Run PID expert and collect demonstrations
2. **Stage 2:** Train imitation learning model
3. **Stage 3:** Test learned policy
4. **Stage 4:** Implement PPO for disturbance recovery

Good luck with your PhD research! 🚁🎓

---

**Document Version:** 1.0
**Last Updated:** 2025-01-16
**Tested On:** Windows 11, WSL2 Ubuntu 22.04, ArduPilot Master