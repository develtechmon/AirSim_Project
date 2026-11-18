r"""
radio_pilot.py - SIMULATOR ONLY: YOUR real Bumblebee radio flies the simulated drone (instead of sim_pilot.py)
=====================================================================================================
sim_pilot.py is a robot that presses the switches for you. This script lets YOU press them: it reads your
radio over USB (the radio acts as a game joystick) and sends your sticks, switches and buttons to the simulator
as RC channels. hit_real.py (and throw_real.py) cannot tell the difference - they see CH8, CH9, CH11-CH16
exactly as they will on the real drone. Practice the whole hit mechanism here first.
Never connect it to the real drone: it sends RC overrides and SIM_ parameters (the real FC has no SIM_ params).

1. FIRST TIME ONLY  (one-time setup, about 5 minutes)
------------------------------------------------------------------------------------------------------
    | Step | What                                                                                      |
    |------|-------------------------------------------------------------------------------------------|
    | 1    | PowerShell:   pip install pygame                                                          |
    | 2    | Radio: plug in the USB cable. When the radio asks for the USB mode choose                 |
    |      | "USB Joystick (HID)". (Menu names differ a little between radios.)                       |
    | 3    | Radio: switch BOTH RF modules OFF (the radio must not transmit while it is a joystick).   |
    |      | Use your normal model (same mixer, same CH8 / CH9 / buttons as on the real drone).        |
    |      | Classic joystick mode (the default) is fine: CH1-CH8 = axes, CH9 and up = buttons.        |
    | 4    | Check Windows sees it: press Win+R, type  joy.cpl  -> your radio is listed, move a stick. |
    | 5    | py .\radio_pilot.py map     (it asks you to move each control once - see section 2)      |
    | 6    | py .\radio_pilot.py show    (check every number looks right - see section 2)             |

2. HOW TO RUN
------------------------------------------------------------------------------------------------------
    | Command                        | What it does                                | Radio needed |
    |--------------------------------|---------------------------------------------|--------------|
    | py .\radio_pilot.py map        | learns which joystick axis / button is each | yes          |
    |                                | radio channel, saves radio_map.json         |              |
    | py .\radio_pilot.py show       | live table of your channels (no simulator)  | yes          |
    | py .\radio_pilot.py fly        | YOU fly the simulator                       | yes          |

    Flying session = 3 windows, same as sim_pilot.py (Throw & Fly guide B3 for window 1):

    Window 1 - WSL: the simulator (MAVProxy holds port 5760; hit_real uses 5762, this script 5763):

    cd ~/sitl_q250_throw

    sim_vehicle.py -v ArduCopter -f quad --model quad:@ROMFS/models/q250.json --console --custom-location=37.6188,-122.3754,5,0 -A "--fg 172.28.0.1"

    Window 2 - PowerShell, folder Q250_Hardware_Throw_Fly: hit_real, wait for "Pi READY":

    py .\hit_real.py --conn tcp:127.0.0.1:5762 --baud 115200

    Window 3 - PowerShell, same folder: your radio:

    py .\radio_pilot.py fly

    Start with: throttle stick DOWN, CH9 (arm) OFF, CH8 on AltHold. The script waits until the throttle is low.
    Never run sim_pilot.py at the same time (two pilots would fight over the same channels).

    Your practice order (same as the field plan):
    | #  | You do                                           | You should see                              |
    |----|--------------------------------------------------|---------------------------------------------|
    | 1  | CH9 arm, CH8 AltHold, throttle up to ~1 m        | FC: ARMED ALT_HOLD, height 1 m              |
    | 2  | CH8 to PosHold                                   | FC: ... POSHOLD                             |
    | 3  | climb to 10-12 m (throttle up, then middle)      | height 10-12 m, hovering                    |
    | 4  | press a RIGHT button (CH14 / CH15 / CH16)        | hit_real window: level + 1/2/3 high beeps   |
    | 5  | press a LEFT button (CH11 flip / CH12 bird /     | hit_real: RECOVERED, back to you in POSHOLD |
    |    | CH13 spin) once                                  |                                             |
    | 6  | press CH13 again, flick CH8 1 s later            | "you took over" - the script lets go        |
    | 7  | press CH13 below 8 m                             | REFUSED (height < 8 m), nothing happens     |
    | 8  | land in PosHold, CH9 off                         | FC: disarmed                                |

3. EXPECTED RESULT
------------------------------------------------------------------------------------------------------
    | Command | Pass when                                                                               |
    |---------|-----------------------------------------------------------------------------------------|
    | map     | every channel line ends "-> axis N" or "-> button N"; "Saved radio_map.json"            |
    | show    | sticks move 1000-2000 (middle ~1500, throttle low ~1000); a pressed button shows        |
    |         | "CH13 ON (2000)"; CH8 gives your 3 or 6 positions (same numbers as Mission Planner)     |
    | fly     | "FC: ARMED ..." when you flip CH9; every switch / button event is printed with its      |
    |         | channel number and a short line "mode ... height ..." updates while you fly            |

4. OUTPUTS
------------------------------------------------------------------------------------------------------
    | File                                         | Content                                          |
    |----------------------------------------------|--------------------------------------------------|
    | radio_map.json                               | which axis / button is which channel (map)       |
    | runs/radio_pilot/radio_<date>_<time>.csv     | every switch / button event with time, mode, hgt |

5. GOOD TO KNOW
------------------------------------------------------------------------------------------------------
    | Item                    | Detail                                                                   |
    |-------------------------|--------------------------------------------------------------------------|
    | Buttons are on / off    | the USB joystick only says "pressed" or "not pressed", so a button is    |
    |                         | sent as 2000 / 1000. The real radio sends 1000-2000; hit_real needs     |
    |                         | above 1700, so 2000 is the same press.                                   |
    | Stop with Ctrl+C        | the simulator then sees "no radio" (RC lost) - land first, then stop.    |
    | Simulator only          | the script refuses any address that is not this PC (127.0.0.1).          |
"""

import argparse
import csv
import json
import math
import os
import sys
import time

os.environ["MAVLINK20"] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")                  # no window needed, only the joystick
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"           # keep reading while the terminal has focus
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(HERE, "radio_map.json")
RUNS = os.path.join(HERE, "runs", "radio_pilot")

# what each channel is for (name shown on screen)
NAMES = {1: "roll", 2: "pitch", 3: "throttle", 4: "yaw", 5: "CH5", 6: "CH6", 7: "CH7 motors off", 8: "CH8 flight mode",
         9: "CH9 arm", 11: "CH11 flip", 12: "CH12 bird", 13: "CH13 spin", 14: "CH14 easy", 15: "CH15 medium",
         16: "CH16 hard"}
REST = {1: 1500, 2: 1500, 3: 1000, 4: 1500, 5: 1000, 6: 1500, 7: 1000, 8: 1300, 9: 1000, 10: 1500,
        11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000}   # used for a channel that is not mapped
AUX = [5, 7, 8, 9, 11, 12, 13, 14, 15, 16]                          # switches / buttons whose changes are printed

# Classic EdgeTX joystick mode: CH1-CH8 = axes 0-7, CH9-CH16 = buttons 0-7. "map" fixes it if your PC orders them differently.
CLASSIC = {str(c): ["axis", c - 1, False] for c in range(1, 9)}
CLASSIC.update({str(c): ["button", c - 9, False] for c in range(9, 17)})

# [channel, text shown, how sure we are about direction: stick = fix the direction, aux = take as it comes]
LEARN = [
    (1, "ROLL stick: push it fully to the RIGHT and hold it there", True),
    (2, "PITCH stick: pull it fully BACK (toward you) and hold it there", True),
    (3, "THROTTLE stick: push it fully UP and hold it there", True),
    (4, "YAW stick: push it fully to the RIGHT and hold it there", True),
    (7, "CH7 (motors off): flip that switch", False),
    (8, "CH8 (flight mode): move that switch through ALL its positions", False),
    (9, "CH9 (arm): flip the arm switch", False),
    (11, "CH11 (left button, flip): press and hold it", False),
    (12, "CH12 (left button, bird): press and hold it", False),
    (13, "CH13 (left button, spin): press and hold it", False),
    (14, "CH14 (right button, easy): press and hold it", False),
    (15, "CH15 (right button, medium): press and hold it", False),
    (16, "CH16 (right button, hard): press and hold it", False),
    (5, "CH5 (throw switch, only for throw_real - do nothing for 12 s to skip it): flip it", False),
]


class Radio:
    """Reads the radio as a joystick and turns it into RC channel numbers (1000-2000)."""

    def __init__(self, index=0, need_map=True):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            sys.exit("No joystick found. Check: radio plugged in, USB mode = USB Joystick (HID), RF modules OFF, "
                     "and the radio shows in Windows (Win+R, joy.cpl).")
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        print(f"Radio found: {self.js.get_name()}  ({self.js.get_numaxes()} axes, {self.js.get_numbuttons()} buttons)")
        self.map, self.from_file = dict(CLASSIC), False
        if os.path.exists(MAP_FILE):
            try:
                with open(MAP_FILE) as f:
                    self.map.update(json.load(f)["channels"])
                self.from_file = True
            except Exception as e:
                print(f"[map] could not read radio_map.json ({e}) - using the classic guess")
        if need_map and not self.from_file:
            print("[map] no radio_map.json yet - using the classic guess (CH1-8 axes, CH9-16 buttons). "
                  "Run  py .\\radio_pilot.py map  once to make it exact.")

    def read_raw(self):
        self.pg.event.pump()
        return ([self.js.get_axis(i) for i in range(self.js.get_numaxes())],
                [self.js.get_button(i) for i in range(self.js.get_numbuttons())])

    def channels(self, raw=None):
        """-> {channel: pwm}. A channel that is not mapped (or not on this radio) gets its rest value."""
        axes, btns = raw if raw is not None else self.read_raw()
        out = {}
        for ch in range(1, 17):
            m, pwm = self.map.get(str(ch)), None
            if m is not None:
                kind, idx, inv = m
                if kind == "axis" and idx < len(axes):
                    v = -axes[idx] if inv else axes[idx]
                    pwm = int(round(min(2000, max(1000, 1500 + 500 * v))))
                elif kind == "button" and idx < len(btns):
                    pwm = 2000 if btns[idx] else 1000
            out[ch] = REST[ch] if pwm is None else pwm
        return out


# ------------------------------------------------------------------------------------------ map
def avg_raw(radio, sec=0.4):
    n, a0, b0, t = 0, None, None, time.time()
    while time.time() - t < sec or n == 0:
        a, b = radio.read_raw()
        if a0 is None:
            a0, b0 = [0.0] * len(a), [0.0] * len(b)
        a0 = [x + y for x, y in zip(a0, a)]
        b0 = [x + y for x, y in zip(b0, b)]
        n += 1
        time.sleep(0.02)
    return [x / n for x in a0], [x / n for x in b0]


def do_map(radio):
    print("\nI will ask you to move one control at a time. Do what it says, hold it ~1 s, then put it back.\n"
          "Start with: all switches OFF / up as for the real drone, throttle stick DOWN.\n")
    chans, used = {}, set()
    for ch, text, stick in LEARN:
        a0, b0 = avg_raw(radio)
        print(f"CH{ch}: {text}", flush=True)
        t0, best, trig = time.time(), {}, None
        while time.time() - t0 < (30 if ch != 5 else 12):
            a, b = radio.read_raw()
            for i, v in enumerate(a):
                if ("axis", i) not in used and i < len(a0):
                    d = v - a0[i]
                    if abs(d) > (0.5 if stick else 0.3):
                        trig = trig or time.time()
                    if abs(d) > abs(best.get(("axis", i), 0)):
                        best[("axis", i)] = d
            for i, v in enumerate(b):
                if ("button", i) not in used and i < len(b0) and abs(v - b0[i]) > 0.5:
                    trig = trig or time.time()
                    best[("button", i)] = float(v - b0[i])
            if trig and time.time() - trig > 0.6:
                break
            time.sleep(0.02)
        cand = {k: d for k, d in best.items() if abs(d) > (0.5 if stick else 0.3)}
        if not cand:
            print(f"   CH{ch}: nothing moved - skipped" + ("" if ch == 5 else "  (run map again if you need it)"))
            continue
        key = max(cand, key=lambda k: abs(cand[k]))
        inv = bool(stick and key[0] == "axis" and cand[key] < 0)
        chans[str(ch)] = [key[0], key[1], inv]
        used.add(key)
        print(f"   CH{ch} -> {key[0]} {key[1]}" + ("  (direction flipped)" if inv else ""))
        print("   now put it back...", flush=True)
        time.sleep(1.8)
    if not chans:
        sys.exit("Nothing was learned. Is the radio in USB Joystick mode with the RF modules off?")
    out = dict(CLASSIC)
    out.update(chans)
    with open(MAP_FILE, "w") as f:
        json.dump({"radio": radio.js.get_name(), "channels": out}, f, indent=1)
    print(f"\nSaved {MAP_FILE}\nNext:  py .\\radio_pilot.py show")


# ------------------------------------------------------------------------------------------ show / events
def quant(radio, ch, pwm):
    m = radio.map.get(str(ch))
    if m and m[0] == "button":
        return "ON" if pwm > 1700 else "off"
    return int(round(pwm / 50.0) * 50)


def label(radio, ch, pwm):
    m = radio.map.get(str(ch))
    if m and m[0] == "button":
        return f"{'ON' if pwm > 1700 else 'off'}   ({pwm})"
    return f"{pwm}"


def clear():
    sys.stdout.write("\r" + " " * 110 + "\r")


def do_show(radio):
    print("\nLive view. Move every control and read the numbers. Ctrl+C = stop.\n"
          "Buttons must show ON (2000) when pressed; throttle low must be about 1000; CH8 gives your positions.\n")
    last = {}
    try:
        while True:
            ch = radio.channels()
            for c in AUX:
                q = quant(radio, c, ch[c])
                if last.get(c) != q:
                    if c in last:
                        clear()
                        print(f"{NAMES[c]:<18} {label(radio, c, ch[c])}")
                    last[c] = q
            sys.stdout.write(f"\rroll {ch[1]:4d}  pitch {ch[2]:4d}  throttle {ch[3]:4d}  yaw {ch[4]:4d}   ")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped")


# ------------------------------------------------------------------------------------------ fly
def do_fly(radio, args):
    if not any(h in args.conn for h in ("127.0.0.1", "localhost")):
        sys.exit("REFUSED: radio_pilot.py is for the simulator only (address must be this PC, 127.0.0.1).")
    from pymavlink import mavutil
    from sim_pilot import YOUR_VALUES                                  # the same drone values as sim_pilot.py

    m = mavutil.mavlink_connection(args.conn, source_system=255)
    m.wait_heartbeat()
    m.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, 33, 100000, 0, 0, 0, 0, 0)
    st = {"mode": "", "armed": False, "alt": 0.0, "start": None, "dist": 0.0}

    def send(ch):
        m.mav.rc_channels_override_send(1, 1, *[ch[i] for i in range(1, 17)], 0, 0)

    def pump():
        while True:
            h = m.recv_match(blocking=False)
            if h is None:
                return
            ty = h.get_type()
            if ty == "HEARTBEAT" and h.get_srcSystem() == 1 and h.type != 6:
                a, md = bool(h.base_mode & 128), mavutil.mode_string_v10(h)
                if (a, md) != (st["armed"], st["mode"]):
                    clear()
                    print(f"{time.time() - t0:6.1f}s  FC: {'ARMED' if a else 'disarmed'} {md}")
                    st["armed"], st["mode"] = a, md
                    log("fc", "", "")
                st["armed"], st["mode"] = a, md
            elif ty == "GLOBAL_POSITION_INT":
                st["alt"] = h.relative_alt / 1000
                if st["start"] is None and h.lat != 0:                  # the spot where you started = the middle of the field
                    st["start"] = (h.lat / 1e7, h.lon / 1e7)
                if st["start"] is not None:
                    dn = (h.lat / 1e7 - st["start"][0]) * 111320.0
                    de = (h.lon / 1e7 - st["start"][1]) * 111320.0 * math.cos(math.radians(st["start"][0]))
                    st["dist"] = math.hypot(dn, de)

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, time.strftime("radio_%Y%m%d_%H%M%S.csv"))
    fh = open(path, "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["time_s", "event", "channel", "value", "fc_mode", "armed", "height_m"])
    t0 = time.time()

    def log(ev, ch, val):
        wr.writerow([f"{time.time() - t0:.2f}", ev, ch, val, st["mode"], int(st["armed"]), f"{st['alt']:.2f}"])
        fh.flush()

    ch = radio.channels()
    if ch[3] > 1100:
        print(f"Throttle stick is up ({ch[3]}). Pull it DOWN to start (safety habit: start with throttle low)...")
        while radio.channels()[3] > 1100:
            time.sleep(0.1)

    if not args.no_setup:
        print("Setting YOUR drone's values in the SIMULATOR: " + ", ".join(f"{k} {v}" for k, v in YOUR_VALUES.items()))
        for k, v in YOUR_VALUES.items():
            m.mav.param_set_send(1, 1, k.encode(), float(v), 9)
        t = time.time()
        while time.time() - t < 3:
            send(radio.channels()); pump(); time.sleep(0.02)

    print("\nYOU are the pilot now. Ctrl+C = stop (land first). Events are printed below.\n")
    last, t_stat = {c: quant(radio, c, radio.channels()[c]) for c in AUX}, 0.0
    try:
        while True:
            ch = radio.channels()
            send(ch)
            pump()
            for c in AUX:
                q = quant(radio, c, ch[c])
                if q != last[c]:
                    clear()
                    print(f"{time.time() - t0:6.1f}s  {NAMES[c]:<18} {label(radio, c, ch[c])}")
                    log("switch", c, ch[c])
                    last[c] = q
            if time.time() - t_stat > 0.4:
                need = []                                               # the same 3 checks hit_real makes with your hands
                if st["mode"] not in ("POSHOLD", "LOITER"):
                    need.append("mode POSHOLD")
                if st["alt"] < 8.0:
                    need.append("climb above 8 m")
                if not 1400 <= ch[3] <= 1600:
                    need.append("throttle DOWN to middle" if ch[3] > 1600 else "throttle UP to middle")
                if st["dist"] > 5.0:
                    need.append(f"fly back to the start spot ({st['dist']:.0f} m away)")
                msg = "HIT OK - press a left button" if not need and st["armed"] else ("on the ground" if not st["armed"] else "to hit: " + ", ".join(need))
                sys.stdout.write(f"\r  {st['mode']:<9} h {st['alt']:5.1f} m  thr {ch[3]:4d}  start {st['dist']:4.1f} m | {msg:<70}")
                sys.stdout.flush()
                t_stat = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        clear()
        print(f"stopped. Event list saved: {path}")
        if st["armed"]:
            print("NOTE: the simulated drone is still armed and now has no radio (RC lost) - the simulator will land/failsafe it.")
    finally:
        fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["map", "show", "fly"])
    ap.add_argument("--conn", default="tcp:127.0.0.1:5763", help="SITL SERIAL2 (MAVProxy has 5760, hit_real 5762)")
    ap.add_argument("--stick", type=int, default=0, help="joystick number if the PC has more than one (default 0)")
    ap.add_argument("--no-setup", action="store_true", help="fly: do not set the drone values (already set)")
    a = ap.parse_args()
    r = Radio(a.stick, need_map=(a.what != "map"))
    {"map": lambda: do_map(r), "show": lambda: do_show(r), "fly": lambda: do_fly(r, a)}[a.what]()
