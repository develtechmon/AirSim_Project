r"""
STAGE 3 FINAL TEST - PPO vs BC vs RULE vs ARDUPILOT  (step 3d, ArduPilot SITL + AirSim)
======================================================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running)
------------------------------------------------------------------------
    Needs models\stage3_final\gated_curriculum_policy.zip = the FROZEN final model (Round 1, see
    train_stage_3_gated_curriculum.py, FINAL METHOD). Yaw brake ON (0.5 s) by default.
    and models\recovery_policy_best.pth + models\obs_norm.npz (train_imitation.py).

    Full thesis test - 27 hits x 5 recoveries = 135 flights (~45 min):

    py .\test_stage_3_gated_curriculum.py

    Quick test - only hard flips (~8 min):

    py .\test_stage_3_gated_curriculum.py --types flip --levels hard

    Test another model (e.g. a checkpoint during training):

    py .\test_stage_3_gated_curriculum.py --model .\models\stage3_checkpoints\gated_checkpoints\gated_curriculum_100000_steps.zip

    More hits for tighter numbers (5 per type and level = 225 flights, ~75 min):

    py .\test_stage_3_gated_curriculum.py --repeats 5

    YAW BRAKE (added 22 Oct 2026, ON by default; see impact_check.py section 9).
    Old behaviour, to repeat a test from before 22 Oct exactly:

    py .\test_stage_3_gated_curriculum.py --model .\models\round1_final.zip --types flip --levels hard --repeats 5 --modes ppo-policy bc-policy attitude-rule spin-stop-fast --yaw-brake 0


2. EXPECTED RESULT
------------------------------------------------------------------------
    Printed tables (and saved), per recovery and level:
        Recovery        Level    Recovered  Avg lowest  Avg loss  Avg time  Below 5 m
        ppo-policy      HARD       100 %      >= 17 m    <= 13 m    ...         0      <- goal
        bc-policy       HARD       100 %      ~16.5 m    ~13.5 m    ...         0      (bc_impact_check)
        attitude-rule   HARD       100 %      ~16.4 m    ...                    0
        spin-stop-fast  HARD       100 %      ~15.3 m    ...                    0      (bar to beat)
        ardupilot       HARD       FAILS often at 3 turns (stock ArduPilot)
    plus "PPO vs ..." lines: height gained and time saved against each baseline.

    How to read it / what to do:
    Result                                         Meaning                               For the thesis
    ppo higher lowest + faster than spin-stop-fast PPO beats ArduPilot's best recovery    main claim proven
    ppo ~= bc-policy                               PPO kept BC but did not improve       train longer / tune
    ppo below 5 m on some hits                     PPO unsafe on those hits              send me the CSVs


3. OUTPUT FILES
------------------------------------------------------------------------
    runs\test_stage_3\test_stage_3_<date>.csv / _steps.csv / .json   every flight (impact_check format)
    analysis_results\test_stage3_<date>.json        the tables below, for the thesis
    analysis_results\table_5_4_recovery_comparison.tex
    analysis_results\test_stage3_<date>_lowest.png  bar chart: avg lowest height per level and recovery


4. WHAT THE SCRIPT DOES
------------------------------------------------------------------------
    Step 1  load the PPO and BC networks BEFORE takeoff
    Step 2  run impact_check: every hit is flown once by EACH recovery (same seed = same hit)
    Step 3  summarise per recovery x level and per recovery x type, compare PPO with each baseline


5. THE RECOVERIES COMPARED
------------------------------------------------------------------------
    ppo-policy      the Stage 3 PPO network (--model)             = THESIS RESULT
    bc-policy       the BC network PPO started from               = what PPO added on top of BC
    attitude-rule   the hand-made rule BC copied                  = the teacher
    spin-stop-fast  ArduPilot with rotation limits off until slow = ArduPilot's best (bar to beat)
    ardupilot       stock ArduPilot                               = what a normal drone does


6. SAME RULES AS impact_check.py
------------------------------------------------------------------------
    hits, curriculum levels, RECOVERED / FAILED rules, 5 m virtual ground, rescue and parameter
    safety are identical - this script only chooses the recoveries and summarises the results.
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

import impact_check as ic

DEFAULT_MODEL = "models/stage3_final/gated_curriculum_policy.zip"   # frozen final model (Round 1)
MODES = ["ppo-policy", "bc-policy", "attitude-rule", "spin-stop-fast", "ardupilot"]
LEVELS = ["easy", "medium", "hard"]
OUT = Path("analysis_results")


def summarise(impacts, modes):
    """Per recovery x level and per recovery x type: recovered %, avg lowest, avg loss, avg time, below 5 m."""
    def stats(rows):
        rec = [r for r in rows if r["recovered"]]
        return {"flights": len(rows),
                "recovered_pct": 100.0 * len(rec) / len(rows),
                "avg_lowest_m": float(np.mean([r["min_alt_m"] for r in rows])),
                "worst_lowest_m": float(np.min([r["min_alt_m"] for r in rows])),
                "avg_height_loss_m": float(np.mean([r["alt_loss_m"] for r in rows])),
                "avg_recovery_time_s": float(np.mean([r["recovery_time_s"] for r in rec])) if rec else None,
                "below_5m": int(sum(bool(r["virtual_ground"]) for r in rows))}
    out = {"by_level": {}, "by_type": {}, "all": {}}
    for m in modes:
        rows = [r for r in impacts if r["recovery"] == m]
        if not rows:
            continue
        out["all"][m] = stats(rows)
        for lv in LEVELS:
            sub = [r for r in rows if r["level"] == lv]
            if sub:
                out["by_level"].setdefault(m, {})[lv] = stats(sub)
        for k in sorted({r["type"] for r in rows}):
            out["by_type"].setdefault(m, {})[k] = stats([r for r in rows if r["type"] == k])
    return out


def print_tables(res, modes, main_mode):
    fmt = lambda v, f: "-" if v is None else format(v, f)
    print("\n" + "=" * 92)
    print("RESULTS PER LEVEL  (start 30 m, virtual ground 5 m; higher lowest / lower loss / lower time = better)")
    print("=" * 92)
    print(f"{'Recovery':<16}{'Level':<8}{'Flights':>8}{'Recovered':>11}{'Avg lowest':>12}{'Worst':>8}"
          f"{'Avg loss':>10}{'Avg time':>10}{'Below 5 m':>11}")
    for m in modes:
        for lv in LEVELS:
            s = res["by_level"].get(m, {}).get(lv)
            if s:
                print(f"{m:<16}{lv.upper():<8}{s['flights']:>8}{s['recovered_pct']:>10.0f}%{s['avg_lowest_m']:>10.1f} m"
                      f"{s['worst_lowest_m']:>6.1f} m{s['avg_height_loss_m']:>8.1f} m"
                      f"{fmt(s['avg_recovery_time_s'], '.2f'):>8} s{s['below_5m']:>11}")
        print()
    print("RESULTS PER TYPE")
    for m in modes:
        for k, s in res["by_type"].get(m, {}).items():
            print(f"{m:<16}{k:<8}{s['flights']:>8}{s['recovered_pct']:>10.0f}%{s['avg_lowest_m']:>10.1f} m"
                  f"{s['worst_lowest_m']:>6.1f} m{s['avg_height_loss_m']:>8.1f} m"
                  f"{fmt(s['avg_recovery_time_s'], '.2f'):>8} s{s['below_5m']:>11}")
    if main_mode in res["all"]:
        p = res["all"][main_mode]
        print("\n" + "=" * 92)
        print(f"{main_mode.upper()} vs EACH BASELINE  (all hits; + = {main_mode} better)")
        print("=" * 92)
        res["comparison"] = {}
        for m in modes:
            if m == main_mode or m not in res["all"]:
                continue
            b = res["all"][m]
            d_low = p["avg_lowest_m"] - b["avg_lowest_m"]
            d_t = (b["avg_recovery_time_s"] - p["avg_recovery_time_s"]
                   if p["avg_recovery_time_s"] is not None and b["avg_recovery_time_s"] is not None else None)
            res["comparison"][m] = {"height_gained_m": d_low, "time_saved_s": d_t,
                                    "recovered_pct_diff": p["recovered_pct"] - b["recovered_pct"]}
            print(f"   vs {m:<16} lowest {d_low:+.1f} m | time {fmt(d_t, '+.2f')} s | recovered "
                  f"{p['recovered_pct'] - b['recovered_pct']:+.0f} points")


def save_outputs(res, modes, stamp, tag, tex_name, caption):
    OUT.mkdir(exist_ok=True)
    jp = OUT / f"{tag}_{stamp}.json"
    json.dump(res, open(jp, "w"), indent=2)
    tp = OUT / tex_name
    with open(tp, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\caption{" + caption + "}\n")
        f.write("\\begin{tabular}{llccccc}\n\\hline\n")
        f.write("Recovery & Level & Recovered (\\%) & Avg lowest (m) & Avg loss (m) & Avg time (s) & Below 5 m \\\\\n\\hline\n")
        for m in modes:
            for lv in LEVELS:
                s = res["by_level"].get(m, {}).get(lv)
                if s:
                    t = "-" if s["avg_recovery_time_s"] is None else f"{s['avg_recovery_time_s']:.2f}"
                    f.write(f"{m} & {lv.upper()} & {s['recovered_pct']:.0f} & {s['avg_lowest_m']:.1f} & "
                            f"{s['avg_height_loss_m']:.1f} & {t} & {s['below_5m']} \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n\\end{table}\n")
    pp = OUT / f"{tag}_{stamp}_lowest.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        present = [m for m in modes if m in res["by_level"]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.8 / max(1, len(present))
        for i, m in enumerate(present):
            vals = [res["by_level"][m].get(lv, {}).get("avg_lowest_m", np.nan) for lv in LEVELS]
            ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, w, label=m)
        ax.axhline(ic.FLOOR, color="red", linestyle="--", linewidth=1, label="virtual ground 5 m")
        ax.set_xticks(range(3))
        ax.set_xticklabels([lv.upper() for lv in LEVELS])
        ax.set_ylabel("average lowest height after the hit (m)")
        ax.set_title(caption)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(pp, dpi=150)
        plt.close(fig)
    except ImportError:
        pp = None
    print(f"\nSaved: {jp}\n       {tp}" + (f"\n       {pp}" if pp else ""))


def run(modes, types, levels, repeats, seed, conn, out_name):
    """Run impact_check with these modes; returns the impacts from its JSON."""
    ic.OUT = Path("runs") / out_name
    ic.load_policies([m for m in modes if m in ic.POLICIES])
    before = set(ic.OUT.glob(f"{out_name}_*.json")) if ic.OUT.exists() else set()
    ic.main(argparse.Namespace(types=types, levels=levels, repeats=repeats, seed=seed, keep_limits=False,
                               compare_recovery=True, modes=modes, conn=conn))
    new = sorted(set(ic.OUT.glob(f"{out_name}_*.json")) - before)
    if not new:
        raise SystemExit("impact_check saved no results")
    return json.load(open(new[-1]))["impacts"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Stage 3 PPO model (.zip)")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(ic.DISTURBANCES))
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--yaw-brake", type=float, default=ic.YAW_BRAKE_T,
                    help="yaw brake time constant (s); 0 = old behaviour, as in the tests before 22 Oct 2026")
    a = ap.parse_args()
    ic.YAW_BRAKE_T = a.yaw_brake
    print(f"   Yaw brake: {'ON, ' + str(a.yaw_brake) + ' s' if a.yaw_brake > 0 else 'OFF (old behaviour)'}")
    if "ppo-policy" in a.modes:
        if not Path(a.model).exists():
            raise SystemExit(f"No {a.model} - train first (train_stage_3_gated_curriculum.py) or pass --model")
        ic.register_policy("ppo-policy", a.model)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    impacts = run(a.modes, a.types, a.levels, a.repeats, a.seed, a.conn, "test_stage_3")
    res = summarise(impacts, a.modes)
    res["settings"] = {"model": a.model, "modes": a.modes, "types": a.types, "levels": a.levels,
                       "repeats": a.repeats, "seed": a.seed, "yaw_brake_t": a.yaw_brake}
    print_tables(res, a.modes, "ppo-policy")
    save_outputs(res, a.modes, stamp, "test_stage3", "table_5_4_recovery_comparison.tex",
                 "Recovery comparison on identical hits (start 30 m)")
