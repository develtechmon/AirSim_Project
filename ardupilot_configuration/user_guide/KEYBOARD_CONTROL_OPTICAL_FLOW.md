# Keyboard Drone Control with Optical Flow

Indoor drone control using MTF-02 optical flow sensor and keyboard input.

## Controls

```
        W            Forward
      A S D          Left / Back / Right

      Q   E          Yaw Left / Yaw Right
      R   F          Up / Down

        T            Takeoff
        L            Land
        G            GUIDED mode
        H            Hover (stop)

      SPACE          Emergency Stop
       ESC           Quit
```

## Requirements

### Python Packages
```bash
pip install dronekit pymavlink pynput
```

### Hardware
- Flight controller with ArduPilot
- MTF-02 optical flow + lidar sensor
- Companion computer (for real flight) or SITL (for simulation)

---

## ArduPilot Parameter Configuration

Configure these parameters in **Mission Planner** (Config → Full Parameter List) before flight.

### Serial Port (assuming TELEM2 / Serial2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SERIAL2_BAUD` | 115 | 115200 baud |
| `SERIAL2_OPTIONS` | 1024 | Don't forward MAVLink to GCS |
| `SERIAL2_PROTOCOL` | 1 | MAVLink protocol |

> **Note:** Change `SERIAL2` to match your connection port (e.g., `SERIAL1`, `SERIAL4`, etc.)

### Optical Flow

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FLOW_TYPE` | 5 | MAVLink optical flow |

### Rangefinder

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RNGFND1_TYPE` | 10 | MAVLink rangefinder |
| `RNGFND1_MAX_CM` | 800 | 8m max range (MTF-02 spec) |
| `RNGFND1_MIN_CM` | 1 | 1cm min range |
| `RNGFND1_ORIENT` | 25 | Downward facing |

### EKF3 Configuration (Indoor Optical Flow)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `AHRS_EKF_TYPE` | 3 | Use EKF3 |
| `EK3_SRC_OPTIONS` | 0 | Default options |
| `EK3_SRC1_POSXY` | 0 | None (integrated from velocity) |
| `EK3_SRC1_POSZ` | 2 | Rangefinder |
| `EK3_SRC1_VELXY` | 5 | Optical Flow |
| `EK3_SRC1_VELZ` | 0 | None |
| `EK3_SRC1_YAW` | 1 | Compass |

### Arming

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ARMING_CHECK` | 14 | Skip GPS check (required for indoor) |

---

## Pre-Flight Checklist

### One-Time Setup
1. [ ] Configure MTF-02 protocol to "mav_apm" using MicoAssistant
2. [ ] Connect MTF-02 to TELEM2 (or other serial port)
3. [ ] Set all parameters above in Mission Planner
4. [ ] Reboot flight controller

### Before Each Flight
1. [ ] Verify in Mission Planner Status tab:
   - `opt_qua` shows optical flow quality (> 0)
   - `rangefinder1` shows height reading
2. [ ] Set EKF Origin:
   - Right-click map → "Set Home Here" → "Set EKF Origin Here"
3. [ ] Test in LOITER mode first (manual control)

---

## Usage

### SITL Simulation
```bash
# Terminal 1: Start ArduPilot SITL
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map

# Terminal 2: Run keyboard control
python keyboard_control.py
```

### Real Hardware
```bash
# Edit CONNECTION_STRING in keyboard_control.py:
# CONNECTION_STRING = '/dev/ttyACM0'  # USB
# CONNECTION_STRING = '/dev/ttyAMA0'  # Raspberry Pi UART

python keyboard_control.py
```

---

## Troubleshooting

### "PreArm: Need Position Estimate"
- EKF origin not set → Set via Mission Planner map
- Optical flow not detected → Check `FLOW_TYPE` and serial config

### "PreArm: Rangefinder not healthy"
- Check `RNGFND1_TYPE = 10`
- Check serial connection
- Sensor too close to ground (< 1cm)

### Can't switch to GUIDED mode
- Try arming in LOITER or ALT_HOLD first
- Check EKF health in Mission Planner

### Drone drifts
- Poor lighting → Need > 60 Lux
- Plain floor → Add texture/patterns
- Calibrate flow → Use `RCx_OPTION = 158`

---

## MTF-02 Specifications

| Spec | Value |
|------|-------|
| TOF Range | 0.02 - 8m |
| Optical Flow FOV | 42° |
| Min Lighting | 60 Lux |
| Output | UART MAVLink |
| Baud Rate | 115200 |
| Update Rate | 100 Hz |
| Weight | ~3g |

---

## Script Configuration

Edit these values in `keyboard_control.py` if needed:

```python
CONNECTION_STRING = 'udp:127.0.0.1:14550'  # SITL default
TAKEOFF_ALTITUDE = 1.0      # meters
DEFAULT_SPEED = 0.5         # m/s
YAW_RATE = 30               # deg/s
VERTICAL_SPEED = 0.3        # m/s
```