# Connection User Guide: AirSim + ArduPilot + DroneKit in WSL2

**Version:** 1.0  
**Last Updated:** November 16, 2025  
**Environment:** Windows 11 + WSL2 Ubuntu + AirSim + ArduPilot SITL

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Network Architecture](#network-architecture)
3. [Prerequisites](#prerequisites)
4. [Connection Setup](#connection-setup)
5. [Troubleshooting](#troubleshooting)
6. [Quick Reference](#quick-reference)

---

## 🎯 System Overview

This guide explains how to connect DroneKit (Python) to ArduPilot SITL when using AirSim for visualization, all running across Windows and WSL2.

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                         WINDOWS                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │              AirSim (Unreal Engine)                │     │
│  │         Receives commands on UDP 9002              │     │
│  │         Sends sensor data on UDP 9003              │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │ Network Bridge                          │
│                    │ (172.23.128.1 ↔ 172.23.143.4)          │
└────────────────────┼─────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────┐
│                    │         WSL2 UBUNTU                      │
│  ┌─────────────────▼──────────────────────────────────┐     │
│  │            ArduPilot SITL                           │     │
│  │  • Connects to AirSim via UDP (172.23.128.1)       │     │
│  │  • Runs flight controller logic                     │     │
│  │  • Outputs MAVLink on TCP 5760                      │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────┐     │
│  │              MAVProxy                               │     │
│  │  • Reads from ArduPilot (TCP 5760)                  │     │
│  │  • Broadcasts telemetry on:                         │     │
│  │    - UDP 172.23.128.1:14550 (to Windows)           │     │
│  │    - UDP 127.0.0.1:14550 (to WSL2 localhost)       │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────┐     │
│  │         Your DroneKit Python Script                 │     │
│  │  • Connects via UDP 127.0.0.1:14550                │     │
│  │  • Sends commands to ArduPilot                      │     │
│  │  • Receives telemetry data                          │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 🌐 Network Architecture

### Understanding the IP Addresses

Think of it like a building with two floors:

- **Windows (172.23.128.1)** = Ground floor, where AirSim lives
- **WSL2 Ubuntu (172.23.143.4)** = Basement floor, where ArduPilot lives
- **Localhost (127.0.0.1)** = "My current floor" (wherever you are)

### Your Current Setup

```
Windows WSL Adapter IP:  172.23.128.1
WSL2 Ubuntu IP:          172.23.143.4
```

**Important:** The WSL2 Ubuntu IP (172.23.143.4) can change after Windows reboot. The Windows adapter IP (172.23.128.1) is usually stable.

### Port Mapping

| Port  | Protocol | Purpose                          | Location |
|-------|----------|----------------------------------|----------|
| 5760  | TCP      | ArduPilot SITL → MAVProxy       | WSL2     |
| 5501  | TCP      | SITL control interface          | WSL2     |
| 9002  | UDP      | ArduPilot → AirSim commands     | Windows  |
| 9003  | UDP      | AirSim → ArduPilot sensor data  | Windows  |
| 14550 | UDP      | MAVProxy telemetry broadcast    | Both     |

---

## ✅ Prerequisites

### 1. Software Installed

**In Windows:**
- AirSim (Unreal Engine environment)
- Windows Subsystem for Linux 2 (WSL2)

**In WSL2 Ubuntu:**
- ArduPilot (installed in `~/ardupilot/`)
- MAVProxy (installed via pip)
- Python 3.8+
- DroneKit (`pip install dronekit`)
- PyMavlink (`pip install pymavlink`)

### 2. AirSim Configuration

Your `settings.json` at `C:\Users\YourUsername\Documents\AirSim\settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1,
  "Vehicles": {
    "Copter": {
      "VehicleType": "ArduCopter",
      "DefaultVehicleState": "Armed",
      "AutoCreate": true,
      "UdpIp": "172.23.128.1",
      "UdpPort": 9003,
      "SitlPort": 9002,
      "UseSerial": false
    }
  }
}
```

**Key Settings:**
- `UdpIp`: Must be your Windows WSL adapter IP (172.23.128.1)
- `UdpPort`: 9003 (AirSim sends sensor data here)
- `SitlPort`: 9002 (AirSim receives commands here)

---

## 🚀 Connection Setup

### Step 1: Start AirSim (Windows)

1. Launch your AirSim environment (Unreal Engine)
2. Wait until you see the drone spawned in the environment
3. Leave AirSim running

### Step 2: Start ArduPilot SITL (WSL2)

Open WSL2 terminal and run:

```bash
cd ~/ardupilot/ArduCopter

sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
  --sim-address=172.23.128.1 \
  --out=127.0.0.1:14550
```

**What Each Flag Does:**

| Flag | Purpose |
|------|---------|
| `-v ArduCopter` | Vehicle type (quadcopter) |
| `-f airsim-copter` | Flight dynamics model for AirSim |
| `--console` | Launch MAVProxy console |
| `--map` | Show map window |
| `--sim-address=172.23.128.1` | Connect to AirSim on Windows |
| `--out=127.0.0.1:14550` | Broadcast telemetry to WSL2 localhost |

**Expected Output:**

```
Starting SITL Airsim
Bind SITL sensor input at 172.23.128.1:9003
AirSim control interface set to 172.23.128.1:9002
Starting sketch 'ArduCopter'
...
Connect tcp:127.0.0.1:5760 source_system=255
Loaded module console
Loaded module map
Telemetry log: mav.tlog
Waiting for heartbeat from tcp:127.0.0.1:5760
MAV> Received 463 parameters (ftp)
```

**Key Indicators of Success:**
- ✅ `Bind SITL sensor input at 172.23.128.1:9003`
- ✅ `AirSim control interface set to 172.23.128.1:9002`
- ✅ `Received 463 parameters`

**Keep this terminal open!**

### Step 3: Connect DroneKit (WSL2)

Open a **second WSL2 terminal** and create your Python script:

```bash
cd ~/your_project_folder
nano dronekit_test.py
```

Paste this code:

```python
from dronekit import connect
import time

# Connect to ArduPilot via MAVProxy
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)

# Check connection
print("=" * 50)
print("CONNECTION SUCCESSFUL!")
print("=" * 50)
print(f"Mode: {vehicle.mode.name}")
print(f"GPS: {vehicle.gps_0.fix_type}")
print(f"Battery: {vehicle.battery.level}%")
print(f"Armed: {vehicle.armed}")
print(f"Position: {vehicle.location.global_frame}")
print("=" * 50)

# Keep connection alive for 10 seconds
time.sleep(10)

# Close
vehicle.close()
print("Connection closed.")
```

Run it:

```bash
python3 dronekit_test.py
```

**Expected Output:**

```
==================================================
CONNECTION SUCCESSFUL!
==================================================
Mode: STABILIZE
GPS: 3
Battery: 100%
Armed: False
Position: LocationGlobal:lat=-35.3632607,lon=149.1652351,alt=583.989990234375
==================================================
Connection closed.
```

---

## 🔧 Troubleshooting

### Problem 1: "Cannot assign requested address"

**Error:**
```
OSError: [Errno 99] Cannot assign requested address
```

**Cause:** You're using the wrong IP address for your current location.

**Solution:**

- If running script in **WSL2**: Use `udp:127.0.0.1:14550`
- If running script in **Windows**: Use `udp:172.23.128.1:14550`

**Analogy:** You can't use your neighbor's address (172.23.128.1) when you're standing in your own house (127.0.0.1).

---

### Problem 2: "No heartbeat in X seconds"

**Error:**
```
WARNING:dronekit:Link timeout, no heartbeat in last 5 seconds
dronekit.APIException: Timeout in initializing connection.
```

**Cause:** DroneKit is listening, but MAVProxy isn't broadcasting to that address.

**Solution:**

Check your `sim_vehicle.py` command includes:
```bash
--out=127.0.0.1:14550
```

Verify MAVProxy output line shows:
```bash
"mavproxy.py" "--out" "172.23.128.1:14550" "--out" "127.0.0.1:14550"
```

**Analogy:** Your phone is working, but nobody's calling. Make sure MAVProxy is broadcasting to your "phone number" (127.0.0.1:14550).

---

### Problem 3: AirSim Not Connecting

**Symptoms:**
- ArduPilot starts but drone doesn't appear in AirSim
- No movement in simulation

**Check:**

1. **Correct IP in settings.json:**
   ```json
   "UdpIp": "172.23.128.1"  // Must be Windows adapter IP
   ```

2. **WSL2 IP hasn't changed:**
   ```bash
   # In WSL2, check your IP
   ip addr show eth0 | grep "inet "
   ```
   
   If it changed from 172.23.143.4 to something else, you might need to update settings.json if you hardcoded the WSL2 IP anywhere (you shouldn't need to).

3. **Firewall blocking:**
   - Windows Firewall might block UDP 9002/9003
   - Temporarily disable to test

---

### Problem 4: Connection Refused (TCP 5760)

**Error in MAVProxy:**
```
[Errno 111] Connection refused
Connect tcp:127.0.0.1:5760 source_system=255
```

**Cause:** ArduCopter binary didn't start properly.

**Solution:**

1. Kill all ArduPilot processes:
   ```bash
   pkill -9 arducopter
   ```

2. Check if port 5760 is in use:
   ```bash
   netstat -tuln | grep 5760
   ```

3. Restart `sim_vehicle.py`

---

### Problem 5: WSL2 IP Changed After Reboot

**Symptom:** Everything worked yesterday, now it doesn't.

**Solution:**

1. Check new WSL2 IP:
   ```bash
   ip addr show eth0 | grep "inet "
   ```

2. If changed, update your `sim_vehicle.py` command if you hardcoded the old IP anywhere (you shouldn't need to for this setup).

3. Windows adapter IP (172.23.128.1) usually stays the same, so your current command should still work:
   ```bash
   sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
     --sim-address=172.23.128.1 \
     --out=127.0.0.1:14550
   ```

---

## 📚 Quick Reference

### Start ArduPilot SITL (The Command You Need)

```bash
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
  --sim-address=172.23.128.1 \
  --out=127.0.0.1:14550
```

### DroneKit Connection String (The Address You Need)

```python
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
```

### Check If ArduPilot Is Running

```bash
ps aux | grep arducopter
```

### Check If Port Is Listening

```bash
netstat -tuln | grep 14550
```

### View MAVProxy Messages

In the MAVProxy terminal, type:
```
status
mode GUIDED
arm throttle
```

---

## 🧠 Understanding the Connection Chain

Think of it like making a phone call through a switchboard:

1. **You (DroneKit)** dial `127.0.0.1:14550`
2. **The switchboard (MAVProxy)** receives your call
3. **Switchboard** forwards it to **ArduPilot SITL** on TCP 5760
4. **ArduPilot** processes your command
5. **ArduPilot** sends control signals to **AirSim** via UDP 9002
6. **AirSim** moves the drone and sends sensor data back via UDP 9003
7. **ArduPilot** receives sensor data and updates its state
8. **MAVProxy** broadcasts the updated state back to you on UDP 14550
9. **You (DroneKit)** receive the telemetry

**This entire loop happens multiple times per second!**

---

## 🎓 Why This Setup Works

### The Wrong Way (What You Initially Tried)

```python
# ❌ WRONG - This tries to connect directly to ArduPilot
vehicle = connect('tcp:127.0.0.1:5760', wait_ready=True)
```

**Problem:** Port 5760 is where **MAVProxy connects TO** ArduPilot, not where you should connect. It's like trying to tap into the internal phone line of a switchboard operator.

### The Right Way (What We're Using)

```python
# ✅ CORRECT - Connect where MAVProxy broadcasts FROM
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
```

**Why:** MAVProxy acts as a "telemetry broadcaster" - it takes data from ArduPilot and sends it out on UDP 14550 where multiple clients can listen. This is the proper way to connect.

**Analogy:** Instead of tapping the switchboard's internal line, you're listening to the public broadcast channel that the switchboard operator transmits on.

---

## 🔍 Common Misconceptions

### Misconception 1: "TCP is more reliable than UDP"

**Reality:** For MAVLink telemetry, UDP is actually better because:
- It allows multiple clients to listen simultaneously
- Dropped packets don't stall the connection
- Lower latency for real-time control

### Misconception 2: "I should connect to port 5760"

**Reality:** Port 5760 is **MAVProxy's input**, not its output. You connect to port 14550 (MAVProxy's output).

**Analogy:** 
- Port 5760 = MAVProxy's **ears** (listening to ArduPilot)
- Port 14550 = MAVProxy's **mouth** (broadcasting to you)

You don't shout into someone's ears - you listen when they speak!

### Misconception 3: "127.0.0.1 and 172.23.128.1 are interchangeable"

**Reality:** These addresses are completely different:
- `127.0.0.1` = "This machine" (wherever you are)
- `172.23.128.1` = Windows WSL adapter (a specific network interface)

From WSL2, you can't bind to 172.23.128.1 because that address belongs to Windows, not WSL2.

---

## 📖 Advanced: Running Script in Windows (Not Recommended)

If you insist on running your Python script in Windows instead of WSL2:

### Step 1: Install DroneKit in Windows

```powershell
pip install dronekit pymavlink
```

### Step 2: Use Windows Connection String

```python
# Connect from Windows to MAVProxy in WSL2
vehicle = connect('udp:172.23.128.1:14550', wait_ready=True)
```

### Why This Isn't Recommended

1. **More complexity:** Adds network boundary issues
2. **Harder debugging:** Errors harder to trace across Windows/WSL2
3. **Less realistic:** Raspberry Pi runs Linux, not Windows
4. **Performance:** Extra network overhead

**Better approach:** Keep everything in WSL2 where it's simpler and closer to your final hardware deployment.

---

## ✅ Verification Checklist

Before troubleshooting, verify each step:

- [ ] AirSim is running in Windows
- [ ] AirSim settings.json has correct `UdpIp: 172.23.128.1`
- [ ] ArduPilot SITL started without errors
- [ ] You see "Bind SITL sensor input at 172.23.128.1:9003"
- [ ] MAVProxy connected (see "Received 463 parameters")
- [ ] MAVProxy shows `--out 127.0.0.1:14550` in startup
- [ ] Your script uses `udp:127.0.0.1:14550`
- [ ] Your script is running in WSL2, not Windows

---

## 🎯 Summary

### The Golden Rules

1. **Run your Python script in WSL2** where ArduPilot lives
2. **Use `udp:127.0.0.1:14550`** to connect
3. **Start SITL with `--out=127.0.0.1:14550`** to broadcast locally
4. **Use `--sim-address=172.23.128.1`** to connect to AirSim in Windows

### The Working Command

```bash
# In WSL2 Terminal 1: Start ArduPilot
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f airsim-copter --console --map \
  --sim-address=172.23.128.1 \
  --out=127.0.0.1:14550

# In WSL2 Terminal 2: Run your script
cd ~/your_project
python3 your_dronekit_script.py
```

### The Working Python Code

```python
from dronekit import connect

vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
print(f"Mode: {vehicle.mode.name}")
vehicle.close()
```

**That's it. This works. Use it.**

---

## 📞 Need Help?

If you're still stuck:

1. **Check the exact error message** - Don't just say "it doesn't work"
2. **Verify each component** using the checklist above
3. **Share your command and output** - Copy-paste the actual terminal output
4. **Check if IPs changed** - Windows reboot can change WSL2 IP

---

**Document Version:** 1.0  
**Tested Configuration:** Windows 11 + WSL2 Ubuntu 22.04 + ArduPilot SITL + AirSim  
**Last Verified:** November 16, 2025