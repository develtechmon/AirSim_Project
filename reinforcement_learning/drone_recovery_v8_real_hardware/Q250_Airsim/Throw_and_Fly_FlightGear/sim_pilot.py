r"""
sim_pilot.py - SIMULATOR ONLY: plays YOUR radio (switches, sticks) and the thrower, so throw_real.py and
hit_real.py can be checked in SITL exactly as they will fly - before the real drone
=====================================================================================================
These are the same pilot steps I used for the 12 Nov 2026 checks ("Q250 Recovery Option B" guide, section 7).
Never connect it to the real drone: it sends RC overrides and SIM_ parameters (the real FC has no SIM_ params).

1. HOW TO RUN  (3 windows; SITL running as in the Throw & Fly guide B3)
------------------------------------------------------------------------------------------------------
    Window 1 - WSL: the simulator (MAVProxy holds port 5760; the script uses 5762, this pilot 5763):

    cd ~/sitl_q250_throw

    sim_vehicle.py -v ArduCopter -f quad --model quad:@ROMFS/models/q250.json --console --custom-location=37.6188,-122.3754,5,0 -A "--fg 172.28.0.1"

    Window 2 - PowerShell, folder Q250_Hardware_Throw_Fly: the script under test, wait for "READY TO USE" / "Pi READY":

    py .\throw_real.py --conn tcp:127.0.0.1:5762 --baud 115200

    py .\hit_real.py --conn tcp:127.0.0.1:5762 --baud 115200

    Window 3 - PowerShell, same folder: the pilot (one of these):

    py .\sim_pilot.py field                          (throw_real: cancel, re-arm, gentle throw, take-over)

    py .\sim_pilot.py gentle                         (throw_real: throw like your throws 1-5, ~5 rad/s)

    py .\sim_pilot.py flip                           (throw_real: fast flip like throw 6, ~15.6 rad/s)

    py .\sim_pilot.py hits                           (hit_real: climb 30 m, CH11 / CH12 / CH13, low test, take-over)

    First it sets YOUR drone's values in the SIMULATOR only (printed): FLTMODE1-6, FLTMODE_CH 8, RC9_OPTION 154,
    MOT_SPIN_ARM 0, MOT_SPOOL_TIME 0.25, RC5_OPTION 0, SIM_BATT_VOLTAGE 15.3, SIM_IMU_POS_Z 0.02 (sensor 2 cm off).

2. EXPECTED RESULT  (pilot window; the script window shows the script's own lines - see the guide, section 7)
------------------------------------------------------------------------------------------------------
    | Command | Pilot window ends with                                    | Pass when                        |
    |---------|-----------------------------------------------------------|----------------------------------|
    | field   | A after cancel: disarmed ALT_HOLD                         | all 4 parts as written; script:  |
    |         | B after CH5 OFF: disarmed ALT_HOLD (READY held 30 s)      | 2x "CH5 OFF - cancelled",        |
    |         | C GUIDED hold lasted 10.1 s / after landing: disarmed     | Throw 1 ... recovered -> GUIDED, |
    |         | D 6 s in PosHold ... height 2.26-2.31 m, mode POSHOLD     | "[stop] you changed the mode to  |
    |         | D after you landed it: disarmed POSHOLD                   | POSHOLD"                         |
    | gentle  | GUIDED hold lasted 10.1 s / after landing: disarmed       | script: "0.02 g < limit 0.41 g at|
    |         |                                                           | spin 4.3", Throw 1 ... caught    |
    |         |                                                           | 0.70 s, lowest about +1.3 m      |
    | flip    | GUIDED hold lasted 10.1 s / after landing: disarmed       | script: "0.60 g < limit 1.12 g at|
    |         |                                                           | spin 15.8", Throw 1 ... caught   |
    |         |                                                           | 0.70 s, lowest about +1.8 m      |
    | hits    | after CH11 / CH12 / CH13: mode POSHOLD, lowest ~18-27 m   | script: 3x "RECOVERED", "back to |
    |         | (--level hard: lost 7.6 / 12.2 / 4.0 m on 12 Nov)         |                                  |
    |         | LOW test at 15 m: mode POSHOLD (refused)                  | you in POSHOLD", REFUSED at 15 m,|
    |         | after take-over: mode POSHOLD                             | "you took over"                  |

3. OUTPUTS
------------------------------------------------------------------------------------------------------
    None of its own: the script under test saves its normal files (runs/throw_real/ or runs/hit_real/).
"""

import argparse
import os
import time

os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

ALTHOLD, POSHOLD = 1300, 1425          # CH8 positions for FLTMODE2 (AltHold) and FLTMODE3 (PosHold)
YOUR_VALUES = {"FLTMODE1": 0, "FLTMODE2": 2, "FLTMODE3": 16, "FLTMODE4": 0, "FLTMODE5": 0, "FLTMODE6": 5,
               "FLTMODE_CH": 8, "RC9_OPTION": 154, "MOT_SPIN_ARM": 0, "MOT_SPOOL_TIME": 0.25, "RC5_OPTION": 0,
               "SIM_BATT_VOLTAGE": 15.3, "SIM_IMU_POS_Z": 0.02}
THROWS = {"up": ((0, 0, -30), (0, 0, 0)),          # field C: straight up
          "gentle": ((0, 0, -30), (0, 20, 0)),     # ~4-5 rad/s pitch, rising ~6 m/s, like throws 1-5
          "flip": ((0, 0, -45), (0, 65, 0))}       # ~15.6 rad/s pitch, like throw 6


class Pilot:
    def __init__(self, conn):
        self.m = mavutil.mavlink_connection(conn, source_system=255)
        self.m.wait_heartbeat()
        self.m.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, 33, 100000, 0, 0, 0, 0, 0)
        self.st = {3: 988, 5: 1000, 8: ALTHOLD, 9: 1000, 11: 1000, 12: 1000, 13: 1000}
        self.mode, self.armed, self.alt, self.t0 = "", False, 0.0, time.time()

    def log(self, s):
        print(f"{time.time() - self.t0:6.1f}s  {s}", flush=True)

    def ps(self, name, v):
        self.m.mav.param_set_send(1, 1, name.encode(), float(v), 9)

    def tick(self, sec):
        t = time.time() + sec
        while time.time() < t:
            s = self.st
            self.m.mav.rc_channels_override_send(1, 1, 1500, 1500, s[3], 1500, s[5], 1500, 1000, s[8], s[9], 1500,
                                                 s[11], s[12], s[13], 0, 0, 0, 0, 0)
            while True:
                h = self.m.recv_match(blocking=False)
                if h is None:
                    break
                ty = h.get_type()
                if ty == "HEARTBEAT" and h.get_srcSystem() == 1 and h.type != 6:
                    a, md = bool(h.base_mode & 128), mavutil.mode_string_v10(h)
                    if (a, md) != (self.armed, self.mode):
                        self.log(f"   FC: {'ARMED' if a else 'disarmed'} {md}")
                    self.armed, self.mode = a, md
                elif ty == "GLOBAL_POSITION_INT":
                    self.alt = h.relative_alt / 1000
            time.sleep(0.05)

    def arm(self):                                   # CH9 switch (RC9_OPTION 154): disarm position first, then arm
        self.st[9] = 1000; self.tick(0.6); self.st[9] = 1900; self.tick(2)
        self.log(f"PILOT CH9 arm -> {'ARMED' if self.armed else 'NOT ARMED'} {self.mode}")
        return self.armed

    def ch5_on(self):
        self.st[5] = 1900; t = time.time()
        while time.time() - t < 3 and self.mode != "GUIDED_NOGPS":
            self.tick(0.1)
        self.tick(0.4); self.log(f"PILOT CH5 ON -> {self.mode} (READY = GUIDED_NOGPS)")

    def throw(self, kind):
        (sx, sy, sz), (tx, ty, tz) = THROWS[kind]
        for n, v in (("SIM_SHOVE_X", sx), ("SIM_SHOVE_Y", sy), ("SIM_SHOVE_Z", sz),
                     ("SIM_TWIST_X", tx), ("SIM_TWIST_Y", ty), ("SIM_TWIST_Z", tz)):
            self.ps(n, v)
        time.sleep(0.2)
        self.ps("SIM_SHOVE_TIME", 300)
        if any((tx, ty, tz)):
            self.ps("SIM_TWIST_TIME", 300)
        self.log(f"THROWER: {kind} throw")

    def watch_landing(self, max_s=60):
        t, tg = time.time(), None
        while self.armed and time.time() - t < max_s:
            self.tick(0.2)
            if self.mode == "GUIDED" and tg is None:
                tg = time.time()
            if self.mode == "LAND" and tg is not None and tg > 0:
                self.log(f"   GUIDED hold lasted {time.time() - tg:.1f} s, height {self.alt:.2f} m"); tg = -1
        self.tick(5); self.log(f"after landing: {'ARMED' if self.armed else 'disarmed'} {self.mode}")
        self.st[5] = 1000; self.tick(1)

    def climb_to(self, alt):
        self.st[8] = POSHOLD; self.st[3] = 1800 if self.alt < alt else 1250; t = time.time()
        while abs(self.alt - alt) > 0.8 and time.time() - t < 60:
            self.tick(0.2)
            if (self.st[3] > 1500 and self.alt > alt) or (self.st[3] < 1500 and self.alt < alt):
                break
        self.st[3] = 1500; self.tick(3); self.log(f"PILOT hovering POSHOLD at {self.alt:.1f} m")

    def hit(self, ch, wait=25):
        self.st[ch] = 1900; self.log(f"PILOT CH{ch} ON"); t, guided, low = time.time(), False, self.alt
        while time.time() - t < wait:
            self.tick(0.1); low = min(low, self.alt)
            guided = guided or self.mode == "GUIDED"
            if guided and self.mode == "POSHOLD":
                break
        self.log(f"   after CH{ch}: mode {self.mode}, height {self.alt:.1f} m, lowest {low:.1f} m")
        self.st[ch] = 1000; self.tick(2)


def main(args):
    p = Pilot(args.conn)
    print("Setting YOUR drone's values in the SIMULATOR: " + ", ".join(f"{k} {v}" for k, v in YOUR_VALUES.items()))
    for k, v in YOUR_VALUES.items():
        p.ps(k, v)
    p.tick(3)
    p.log(f"start: CH8 AltHold position -> {p.mode}")
    if args.what == "field":
        p.log("== A: arm, CH5 ON, cancel with CH5 OFF")
        p.arm(); p.ch5_on(); p.st[5] = 1000; p.tick(5)
        p.log(f"A after cancel: {'ARMED' if p.armed else 'disarmed'} {p.mode}")
        p.log("== B: arm again WITHOUT touching CH8, CH5 ON, no throw for 25 s, CH5 OFF")
        if p.arm():
            p.ch5_on(); p.tick(25); p.st[5] = 1000; p.tick(5)
            p.log(f"B after CH5 OFF: {'ARMED' if p.armed else 'disarmed'} {p.mode}")
        p.log("== C: arm, CH5 ON, throw up, hands off (catch -> hold 10 s -> LAND)")
        if p.arm():
            p.ch5_on(); p.throw("up"); p.watch_landing()
        p.log("== D: arm, CH5 ON, throw, TAKE OVER 0.6 s later: stick MIDDLE, CH8 PosHold")
        if p.arm():
            p.ch5_on(); p.throw("up"); p.tick(0.6)
            p.st[3] = 1500; p.tick(0.15); p.st[8] = POSHOLD; p.tick(1.0)
            hs = []
            for _ in range(12):
                p.tick(0.5); hs.append(p.alt)
            p.log(f"D 6 s in PosHold, stick middle, hands off: height {min(hs):.2f}-{max(hs):.2f} m, mode {p.mode}")
            p.st[3] = 1250; t = time.time()
            while p.alt > 0.15 and time.time() - t < 30:
                p.tick(0.3)
            p.st[3] = 988; t = time.time()
            while p.armed and time.time() - t < 20:
                p.tick(0.3)
            p.st[9] = 1000; p.tick(3)
            p.log(f"D after you landed it: {'ARMED' if p.armed else 'disarmed'} {p.mode} (script must NOT change your mode)")
            p.st[5] = 1000; p.st[8] = ALTHOLD; p.tick(2)
    elif args.what in ("gentle", "flip"):
        p.log(f"== {args.what}: arm, CH5 ON, throw, hands off")
        if p.arm():
            p.ch5_on(); p.throw(args.what); p.watch_landing()
            p.st[9] = 1000; p.tick(2)
    elif args.what == "hits":
        p.log("== hits: take off (CH9 arm, CH8 PosHold, stick up), hover 30 m, then the hit switches")
        p.arm(); p.climb_to(30)
        p.hit(11)
        p.climb_to(30); p.hit(12)
        p.climb_to(30); p.hit(13)
        p.climb_to(15); p.st[11] = 1900; p.tick(3)
        p.log(f"   LOW test CH11 at 15 m: mode {p.mode} (expect POSHOLD = refused)"); p.st[11] = 1000; p.tick(2)
        p.climb_to(30); p.st[11] = 1900; p.log("PILOT CH11 ON, then TAKE OVER 1.0 s later: CH8 AltHold -> PosHold")
        p.tick(1.0); p.st[8] = ALTHOLD; p.tick(0.5); p.st[8] = POSHOLD; p.tick(4)
        p.log(f"   after take-over: mode {p.mode}, height {p.alt:.1f} m"); p.st[11] = 1000; p.tick(2)
        p.log("landing: PosHold, stick low")
        p.st[3] = 1300; t = time.time()
        while p.alt > 0.3 and time.time() - t < 90:
            p.tick(0.3)
        p.st[3] = 988; p.tick(4); p.st[9] = 1000; p.tick(3)
        p.log(f"END: {'ARMED' if p.armed else 'disarmed'} {p.mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["field", "gentle", "flip", "hits"])
    ap.add_argument("--conn", default="tcp:127.0.0.1:5763", help="SITL SERIAL2 (MAVProxy has 5760, the script 5762)")
    main(ap.parse_args())
