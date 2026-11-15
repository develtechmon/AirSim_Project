r"""
BC IMPACT CHECK  (Stage 3 step 3b-check, ArduPilot SITL + AirSim)
=================================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running, ~15 min)
------------------------------------------------------------------------
    Needs models\recovery_policy_best.pth + models\obs_norm.npz from train_imitation.py.

    py .\bc_impact_check.py

    Quick test - only hard flips (~3 min):

    py .\bc_impact_check.py --types flip --levels hard


2. EXPECTED RESULT  (summary table, "Avg low" = average lowest height after the hit, start 30 m)
------------------------------------------------------------------------
    Recovery         Flip EASY  MEDIUM  HARD  | Bird EASY  MEDIUM  HARD  | Spin  | Below 5 m
    bc-policy        target: within ~1 m of attitude-rule on every row                |    0
    attitude-rule      22.7     19.1    16.6  |   27.0     21.9    21.4  |  ~30  |    0   (Test 6)
    spin-stop-fast     22.9     18.4    17.0  |   26.7     24.7    19.8  |  ~30  |    0   (Test 6)

    How to read it / what to do:
    Result                                   Meaning                                Action
    bc-policy ~= attitude-rule, 0 below 5 m  the network can fly the recovery       next: Stage 3 PPO
    bc-policy 2-3 m worse, 0 below 5 m       small copy errors add up in flight     still OK - PPO fixes it
    bc-policy below 5 m / RESCUED often      network lost in unseen situations      send the _steps.csv


3. OUTPUT FILES  (folder runs\bc_impact_check\ - separate from impact_check)
------------------------------------------------------------------------
    bc_impact_check_<date>.csv         one line per impact
    bc_impact_check_<date>_steps.csv   every 50 Hz step (for plots)
    bc_impact_check_<date>.json        settings, limits used, summary


4. WHAT IS DIFFERENT FROM impact_check.py
------------------------------------------------------------------------
                    impact_check.py                         bc_impact_check.py
    Question        "which RULE recovers best?"             "can the trained NETWORK fly it?"
    Recoveries      ardupilot / spin-stop-fast /            bc-policy  vs  attitude-rule (its teacher)
                    rate-rule / attitude-rule ...           vs  spin-stop-fast (bar to beat)
    Needs a model   no                                      yes (models\ from train_imitation.py)
    Output folder   runs\impact_check\                      runs\bc_impact_check\
    Same as before  the hits, levels, pass/fail rules, rescue and safety are IDENTICAL
                    (this script reuses impact_check.py, so both are compared fairly)


5. WHAT bc-policy DOES  (every 20 ms, same as the real drone's Pi will)
------------------------------------------------------------------------
    Step 1  read the 13 inputs from ATTITUDE_QUATERNION + LOCAL_POSITION_NED
    Step 2  scale them with models\obs_norm.npz
    Step 3  network -> roll, pitch, thrust x hover
    Step 4  send SET_ATTITUDE_TARGET (attitude + thrust) -> ArduPilot's 400 Hz loop turns the drone
    Step 5  calm (rate < 1.2 rad/s, tilt < 30 deg, falling < 1 m/s) -> hand back to ArduPilot
"""

import argparse
from pathlib import Path

import impact_check as ic

MODES = ["bc-policy", "attitude-rule", "spin-stop-fast"]

if __name__ == "__main__":
    for f in ("models/recovery_policy_best.pth", "models/obs_norm.npz"):
        if not Path(f).exists():
            raise SystemExit(f"No {f} - run train_imitation.py first")
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(ic.DISTURBANCES))
    ap.add_argument("--levels", nargs="+", default=ic.LEVELS, choices=ic.LEVELS)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    a = ap.parse_args()
    ic.OUT = Path("runs") / "bc_impact_check"
    ic.load_bc()                     # BEFORE takeoff - never load the network mid-recovery
    ic.main(argparse.Namespace(types=a.types, levels=a.levels, repeats=a.repeats, seed=a.seed, keep_limits=False,
                               compare_recovery=True, modes=MODES, conn=a.conn))
