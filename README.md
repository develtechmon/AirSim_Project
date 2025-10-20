# AirSim Python Drone Control Guide

## Quick Start

### 1. Install AirSim Python Package
```bash
pip install airsim
```

### 2. Update settings.json
Make sure API control is allowed:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "FlyWithMe",
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 0,
        "AllowAPIWhenDisconnected": true
      }
    }
  }
}
```

**Important settings:**
- `"AllowAPIAlways": true` - Allows Python to control even without RC connected
- `"AllowAPIWhenDisconnected": true` - Keeps API active if RC disconnects

### 3. Run Your Script
```bash
# Make sure AirSim/Unreal is running first!
python drone_control.py
```

---

## Understanding AirSim Coordinates (NED System)

AirSim uses **NED coordinates**:
- **N** = North (X axis)
- **E** = East (Y axis)  
- **D** = Down (Z axis)

```
        North (+X, Forward)
             ↑
             |
West (-Y) ←--+--→ East (+Y, Right)
             |
             ↓
        South (-X, Backward)

Up = -Z (negative)
Down = +Z (positive)
```

**Examples:**
- Move forward 5m: `x=5, y=0, z=0`
- Move left 3m: `x=0, y=-3, z=0`
- Move up 10m: `x=0, y=0, z=-10`

---

## Key AirSim Functions

### Connection & Setup
```python
import airsim

# Connect to simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control (required!)
client.enableApiControl(True)

# Arm the drone
client.armDisarm(True)
```

### Basic Movement

**1. Takeoff**
```python
client.takeoffAsync().join()  # Goes to ~3m altitude
```

**2. Move to Altitude**
```python
# Move to 5 meters up (z=-5 in NED)
client.moveToZAsync(z=-5, velocity=2).join()
```

**3. Move to Position**
```python
# Move to specific XYZ position
client.moveToPositionAsync(x=10, y=5, z=-10, velocity=3).join()
# This moves to: 10m forward, 5m right, 10m up
```

**4. Move by Velocity**
```python
# Move at constant velocity for duration
client.moveByVelocityAsync(vx=2, vy=0, vz=0, duration=3).join()
# Moves forward at 2 m/s for 3 seconds
```

**5. Move by Velocity with Z control**
```python
# Maintain altitude while moving
client.moveByVelocityZAsync(vx=1, vy=1, z=-5, duration=2).join()
# Moves forward+right while staying at 5m altitude
```

**6. Land**
```python
client.landAsync().join()
```

### Getting Information

**Current Position**
```python
pose = client.simGetVehiclePose()
x = pose.position.x_val
y = pose.position.y_val
z = pose.position.z_val
print(f"Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
```

**Current Velocity**
```python
state = client.getMultirotorState()
vx = state.kinematics_estimated.linear_velocity.x_val
vy = state.kinematics_estimated.linear_velocity.y_val
vz = state.kinematics_estimated.linear_velocity.z_val
print(f"Velocity: {vx:.2f}, {vy:.2f}, {vz:.2f} m/s")
```

**GPS Location**
```python
gps = client.getGpsData()
print(f"GPS: Lat={gps.gnss.geo_point.latitude}, Lon={gps.gnss.geo_point.longitude}")
```

---

## Common Patterns

### Square Flight Pattern
```python
# Takeoff
client.takeoffAsync().join()
client.moveToZAsync(-5, velocity=2).join()

# Get start position
pose = client.simGetVehiclePose()
x0, y0 = pose.position.x_val, pose.position.y_val

# Fly a 5m square
size = 5
client.moveToPositionAsync(x0 + size, y0, -5, 2).join()  # Forward
time.sleep(1)
client.moveToPositionAsync(x0 + size, y0 + size, -5, 2).join()  # Right
time.sleep(1)
client.moveToPositionAsync(x0, y0 + size, -5, 2).join()  # Backward
time.sleep(1)
client.moveToPositionAsync(x0, y0, -5, 2).join()  # Left (back to start)
time.sleep(1)

# Land
client.landAsync().join()
```

### Circle Pattern
```python
import math

# Takeoff
client.takeoffAsync().join()
client.moveToZAsync(-5, velocity=2).join()

# Get center position
pose = client.simGetVehiclePose()
center_x, center_y = pose.position.x_val, pose.position.y_val

# Fly a circle
radius = 5
for angle in range(0, 360, 10):
    rad = math.radians(angle)
    x = center_x + radius * math.cos(rad)
    y = center_y + radius * math.sin(rad)
    client.moveToPositionAsync(x, y, -5, velocity=2).join()
    time.sleep(0.1)

# Land
client.landAsync().join()
```

### Scanning Pattern (Lawn Mower)
```python
# Takeoff
client.takeoffAsync().join()
client.moveToZAsync(-5, velocity=2).join()

# Get start position
pose = client.simGetVehiclePose()
x0, y0 = pose.position.x_val, pose.position.y_val

# Scan back and forth
rows = 5
spacing = 3
length = 15

for i in range(rows):
    # Move forward or backward (alternating)
    if i % 2 == 0:
        client.moveToPositionAsync(x0 + length, y0 + i*spacing, -5, 3).join()
    else:
        client.moveToPositionAsync(x0, y0 + i*spacing, -5, 3).join()
    time.sleep(0.5)

# Return home
client.moveToPositionAsync(x0, y0, -5, 3).join()
client.landAsync().join()
```

---

## Safety Tips

### Always Use Try-Finally
```python
try:
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Your flight code here
    
finally:
    # Always cleanup, even if error occurs
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
```

### Emergency Stop
```python
# In case of emergency, land immediately
client.landAsync().join()

# Or hover in place
client.hoverAsync().join()
```

### Check Connection
```python
# Before flying, verify connection
if not client.ping():
    print("Cannot connect to AirSim!")
    exit()
```

---

## Troubleshooting

**Problem: Script connects but drone doesn't move**
- Solution: Make sure `AllowAPIAlways: true` in settings.json
- Make sure you called `client.enableApiControl(True)`
- Make sure drone is armed: `client.armDisarm(True)`

**Problem: "Drone is not armed" error**
- Solution: Call `client.armDisarm(True)` before takeoff

**Problem: Drone moves but crashes immediately**
- Solution: Increase velocity values (too slow can be unstable)
- Use `.join()` to wait for commands to complete

**Problem: Cannot connect to AirSim**
- Solution: Make sure Unreal/AirSim is running first
- Check if API port is blocked (default: 41451)

---

## Advanced: Async Operations

For faster operations, run commands without waiting:

```python
# Start movement (non-blocking)
task1 = client.moveToPositionAsync(10, 0, -5, 2)

# Do other stuff while moving
print("Drone is moving...")

# Wait for completion when ready
task1.join()
```

---

## Your Specific Task (Simplified)

```python
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Execute your pattern
client.takeoffAsync().join()
client.moveToZAsync(-5, velocity=2).join()  # 5m up
time.sleep(1)

client.moveByVelocityAsync(0, -1, 0, 1).join()  # Left 1m
time.sleep(1)
client.moveByVelocityAsync(0, 1, 0, 1).join()   # Right 1m  
time.sleep(1)
client.moveByVelocityAsync(-1, 0, 0, 1).join()  # Backward 1m
time.sleep(1)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
```

That's it! You're ready to fly with Python!
