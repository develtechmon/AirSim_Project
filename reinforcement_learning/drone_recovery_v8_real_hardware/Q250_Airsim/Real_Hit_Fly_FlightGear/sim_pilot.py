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

    py .\sim_pilot.py hits                           (hit_real b: hover 10 m, CH13 / CH12 / CH11, low test, take-over)

    py .\sim_pilot.py hits --level medium            (hit_real d: presses the RIGHT button CH15 first, then the same hits)

    py .\sim_pilot.py safety                         (throw_real g: the 14 Nov fly-away case - must stay on the ground)

    py .\sim_pilot.py hitsafety                      (hit_real c: breaks the thrust switch / a limit on purpose
                                                      before a hit - guard B must repair it; run this before the field)

    First it sets YOUR drone's values in the SIMULATOR only (printed): FLTMODE1-6, FLTMODE_CH 8, RC9_OPTION 154,
    MOT_SPIN_ARM 0, MOT_SPOOL_TIME 0.25, RC5_OPTION 0, RC7_OPTION 81, SIM_BATT_VOLTAGE 15.3, SIM_IMU_POS_Z 0.02
    (sensor 2 cm off). "safety" (14 Nov) also sets GUID_OPTIONS 8 on purpose and puts it back to 0 at the end.

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
    | safety  | S1 result: highest +0.00 m (switch left ON = 14 Nov)      | script (throw_real g): "[fix]    |
    |         | S2 result: highest +0.00 m (switch ON during READY)       | thrust switch ... read back",    |
    |         | S3 result: highest below 1 m, disarmed after < 1 s        | "[watchdog] ... DISARMED".       |
    |         |                                                           | Version f: +14 to +19 m each     |
    | hits    | after CH13 / CH12 / CH11 (hover 10 m, level easy): mode   | script: 3x "RECOVERED" (or       |
    |         | POSHOLD, lowest ~4-7 m (13 Nov: lost 3.1-6.0 m)           | "rescued"), "back to you in      |
    |         | LOW test at 6 m: mode POSHOLD (refused)                   | POSHOLD", REFUSED at 6 m,        |
    |         | after take-over (CH13 spin): mode POSHOLD                 | "you took over", never above 15 m|

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
               "SIM_BATT_VOLTAGE": 15.3, "SIM_IMU_POS_Z": 0.02, "RC7_OPTION": 81}
THROWS = {"up": ((0, 0, -30), (0, 0, 0)),          # field C: straight up
          "gentle": ((0, 0, -30), (0, 20, 0)),     # ~4-5 rad/s pitch, rising ~6 m/s, like throws 1-5
          "flip": ((0, 0, -45), (0, 65, 0))}       # ~15.6 rad/s pitch, like throw 6


class Pilot:
    def __init__(self, conn):
        self.m = mavutil.mavlink_connection(conn, source_system=255)
        self.m.wait_heartbeat()
        self.m.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, 33, 100000, 0, 0, 0, 0, 0)
        self.st = {3: 988, 5: 1000, 7: 1000, 8: ALTHOLD, 9: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000, 15: 1000, 16: 1000}
        self.mode, self.armed, self.alt, self.t0 = "", False, 0.0, time.time()

    def log(self, s):
        print(f"{time.time() - self.t0:6.1f}s  {s}", flush=True)

    def ps(self, name, v):
        self.m.mav.param_set_send(1, 1, name.encode(), float(v), 9)

    def gp(self, name, timeout=3.0):
        """c: read one parameter back from the FC. Returns the value, or None if it does not answer."""
        self.m.mav.param_request_read_send(1, 1, name.encode(), -1)
        t = time.time() + timeout
        while time.time() < t:
            h = self.m.recv_match(type="PARAM_VALUE", blocking=False)
            if h is not None:
                pid = h.param_id.decode(errors="ignore") if isinstance(h.param_id, bytes) else h.param_id
                if pid.rstrip("\x00") == name:
                    return round(float(h.param_value), 3)
            self.pump()
            time.sleep(0.02)
        return None

    def pump(self):                                  # one radio frame out, all FC messages in
        s = self.st
        self.m.mav.rc_channels_override_send(1, 1, 1500, 1500, s[3], 1500, s[5], 1500, s[7], s[8], s[9], 1500,
                                             s[11], s[12], s[13], s[14], s[15], s[16], 0, 0)
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

    def tick(self, sec):
        t = time.time() + sec
        while time.time() < t:
            self.pump()
            time.sleep(0.05)

    def watch_ground(self, sec, force_thrust=None):
        """Safety tests: watch the height for sec seconds; CH7 (motors off) above 3 m, like you did on 14 Nov.
        force_thrust: also send a 'thrust' command from outside, 200 times a second (a take-off nobody asked for)."""
        t0, top, t_pump, t_dis, a0 = time.time(), -9.0, 0.0, None, self.alt
        while time.time() - t0 < sec:
            if force_thrust is not None and self.armed:
                self.m.mav.set_attitude_target_send(0, 1, 1, 0b00000111, [1, 0, 0, 0], 0, 0, 0, force_thrust)
            if time.time() - t_pump > 0.05:
                self.pump(); t_pump = time.time()
                top = max(top, self.alt - a0)
                if not self.armed and t_dis is None:
                    t_dis = time.time() - t0
                if self.alt - a0 > 3.0 and self.st[7] == 1000:
                    self.st[7] = 1900; self.log(f"   PILOT CH7 (motors off) at {self.alt - a0:.1f} m above the start")
            time.sleep(0.005)
        return top, t_dis

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

    def level(self, name):
        """d: press one RIGHT button (CH14 easy / CH15 medium / CH16 hard) like a thumb press, then let go."""
        ch = {"easy": 14, "medium": 15, "hard": 16}[name]
        self.st[ch] = 1900; self.log(f"PILOT right button CH{ch} -> level {name}"); self.tick(0.6)
        self.st[ch] = 1000; self.tick(1.0)

    def hit(self, ch, wait=60):                      # e: room for the fly-back (up to 30 s)
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
    elif args.what == "safety":
        def reset():
            p.st[5] = 1000; p.tick(4); p.st[7] = 1900; p.tick(1); p.st[7] = 1000; p.st[9] = 1000; p.tick(3)
        p.log("== S1 (= 14 Nov): thrust switch LEFT ON (GUID_OPTIONS 8), arm, CH5 ON, NO throw, watch 8 s")
        p.ps("GUID_OPTIONS", 8); p.tick(1)
        if p.arm():
            p.ch5_on(); top, td = p.watch_ground(8)
            p.log(f"S1 result: highest {top:+.2f} m above the start, mode {p.mode}, {'ARMED' if p.armed else 'disarmed'}")
        reset()
        p.log("== S2: READY first, THEN the thrust switch goes ON (after check 10), NO throw, watch 6 s")
        if p.arm():
            p.ch5_on(); p.tick(2); p.ps("GUID_OPTIONS", 8); top, td = p.watch_ground(6)
            p.log(f"S2 result: highest {top:+.2f} m above the start, mode {p.mode}, {'ARMED' if p.armed else 'disarmed'}")
        reset()
        p.log("== S3: READY, then a take-off FORCED from outside (thrust 60 %, 200 x a second), watch 5 s")
        if p.arm():
            p.ch5_on(); p.tick(1); p.ps("GUID_OPTIONS", 8); top, td = p.watch_ground(5, force_thrust=0.6)
            p.log(f"S3 result: highest {top:+.2f} m above the start, disarmed after "
                  f"{'%.2f s' % td if td is not None else 'NEVER'}, mode {p.mode}")
        reset()
        p.ps("GUID_OPTIONS", 0); p.tick(1)
        p.log(f"END: {'ARMED' if p.armed else 'disarmed'} {p.mode}")
    elif args.what == "hitsafety":
        # c (16 Nov): the same three things that version c adds to hit_real.py, made to happen on purpose.
        p.log("== hitsafety: hover 10 m, then break the thrust switch on purpose before each hit (guard B)")
        p.arm(); p.climb_to(10)
        p.log("== H1: GUID_OPTIONS left at 8 (the 14 Nov value) BEFORE CH13 ON")
        p.ps("GUID_OPTIONS", 8); p.tick(1); p.hit(13)
        p.log("   H1 expect in the script window: '[fix] thrust switch / limits were wrong - fixed and read back'")
        p.log(f"   H1 GUID_OPTIONS now: {p.gp('GUID_OPTIONS')}  (expect 0)")
        p.climb_to(10)
        p.log("== H2: a rotation limit left at 0 BEFORE CH13 ON")
        p.ps("ATC_ACCEL_R_MAX", 0); p.tick(1); p.hit(13)
        p.log(f"   H2 ATC_ACCEL_R_MAX now: {p.gp('ATC_ACCEL_R_MAX')}  (expect 282459)")
        p.climb_to(10)
        p.log("== H3: a normal hit - GUID_OPTIONS must be 8 DURING it and back to 0 after")
        p.hit(13)
        p.log(f"   H3 GUID_OPTIONS after: {p.gp('GUID_OPTIONS')}  (expect 0)")
        p.log("   H3 expect in the script window: '[params] confirmed by the FC' after the hit")
        p.log("landing: PosHold, stick low")
        p.st[3] = 1300; t = time.time()
        while p.alt > 0.3 and time.time() - t < 90:
            p.tick(0.3)
        p.st[3] = 988; p.tick(4); p.st[9] = 1000; p.tick(3)
        p.log(f"END: {'ARMED' if p.armed else 'disarmed'} {p.mode}")
    elif args.what == "hits":
        p.log("== hits: take off (CH9 arm, CH8 PosHold, stick up), hover 10 m, then the hit switches (spin first)")
        p.arm(); p.climb_to(10)
        if args.level:
            p.level(args.level)                          # d: right button first, like on the field
        p.hit(13)
        p.climb_to(10); p.hit(12)
        p.climb_to(10); p.hit(11)
        p.climb_to(6); p.st[13] = 1900; p.tick(3)
        p.log(f"   LOW test CH13 at 6 m: mode {p.mode} (expect POSHOLD = refused)"); p.st[13] = 1000; p.tick(2)
        p.climb_to(10); p.st[13] = 1900; p.log("PILOT CH13 ON, then TAKE OVER 1.0 s later: CH8 AltHold -> PosHold")
        p.tick(1.0); p.st[8] = ALTHOLD; p.tick(0.5); p.st[8] = POSHOLD; p.tick(4)
        p.log(f"   after take-over: mode {p.mode}, height {p.alt:.1f} m"); p.st[13] = 1000; p.tick(2)
        p.log("landing: PosHold, stick low")
        p.st[3] = 1300; t = time.time()
        while p.alt > 0.3 and time.time() - t < 90:
            p.tick(0.3)
        p.st[3] = 988; p.tick(4); p.st[9] = 1000; p.tick(3)
        p.log(f"END: {'ARMED' if p.armed else 'disarmed'} {p.mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["field", "gentle", "flip", "hits", "safety", "hitsafety"])
    ap.add_argument("--level", default=None, choices=["easy", "medium", "hard"],
                    help="hits only (hit_real d): press this RIGHT button before the hits. Leave out = level unchanged")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5763", help="SITL SERIAL2 (MAVProxy has 5760, the script 5762)")
    main(ap.parse_args())
