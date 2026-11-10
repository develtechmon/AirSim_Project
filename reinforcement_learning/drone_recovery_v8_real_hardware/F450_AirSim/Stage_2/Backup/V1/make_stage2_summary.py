"""
MAKE STAGE 2 SUMMARY (ArduPilot)
================================
Builds the Stage 2 executive summary automatically from the saved files of ONE run folder.

Reads:
    runs\\run<N>\\models\\warm|cold\\training_info.json     training settings + transfer check
    runs\\run<N>\\logs\\stage2\\warm|cold\\episodes.csv     learning curves
    runs\\run<N>\\results\\stage2_statistics_<newest>.json  PID / Stage 1 / warm / cold in wind
Writes:
    runs\\run<N>\\results\\STAGE2_SUMMARY_run<N>.md
    runs\\run<N>\\results\\learning_curves_warm_vs_cold.png

Run: py .\\make_stage2_summary.py            (latest run)
     py .\\make_stage2_summary.py --run 1
"""

import argparse
import csv
import datetime
import json
import textwrap
from pathlib import Path

import numpy as np

from drone_hover_disturbance_env import run_folder

ROLL = 20
WRAP = 34           # console table: wrap cells longer than this


def load_json(p):
    return json.load(open(p)) if p and Path(p).exists() else None


def load_rows(p):
    if not Path(p).exists():
        return []
    with open(p) as f:
        return [r for r in csv.DictReader(f) if r["reason"] != "update_pause"]


def learning_stats(rows):
    if not rows:
        return None
    ok = np.array([r["reason"] == "timeout" for r in rows], float)
    t = np.array([int(r["timesteps"]) for r in rows])
    roll = np.convolve(ok, np.ones(ROLL) / ROLL, mode="valid") if len(ok) >= ROLL else np.array([])
    hit = np.where(roll >= 0.8)[0]
    first, last = rows[:50], rows[-50:]
    s = lambda rr: 100.0 * np.mean([r["reason"] == "timeout" for r in rr])
    d = lambda rr: float(np.mean([float(r["mean_dist_m"]) for r in rr]))
    return {"episodes": len(rows), "timesteps": int(t[-1]),
            "first50_success_pct": s(first), "first50_mean_dist_m": d(first),
            "last50_success_pct": s(last), "last50_mean_dist_m": d(last),
            "steps_to_80pct": int(t[hit[0] + ROLL - 1]) if len(hit) else None}


def plot(rows_by, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    drawn = False
    for name, rows in rows_by.items():
        if len(rows) < ROLL:
            continue
        t = np.array([int(r["timesteps"]) for r in rows])
        ok = np.array([100.0 * (r["reason"] == "timeout") for r in rows])
        dist = np.array([float(r["mean_dist_m"]) for r in rows])
        k = np.ones(ROLL) / ROLL
        ax[0].plot(t[ROLL - 1:], np.convolve(ok, k, "valid"), label=f"{name} start", linewidth=1.8)
        ax[1].plot(t[ROLL - 1:], np.convolve(dist, k, "valid"), label=f"{name} start", linewidth=1.8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax[0].set_ylabel(f"success (%), rolling {ROLL}")
    ax[1].set_ylabel(f"mean distance (m), rolling {ROLL}")
    ax[1].set_xlabel("timesteps")
    ax[0].set_title("Stage 2 PPO in wind: warm start (Stage 1) vs cold start (random)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def console_text(lines):
    """The .md file renders in a Markdown viewer; the console gets plain aligned tables."""
    out, block = [], []

    def flush():
        if not block:
            return
        rows = [[c.strip().replace("**", "").replace("`", "") for c in r.strip().strip("|").split("|")]
                for r in block if not set(r.replace("|", "").strip()) <= set("- ")]
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        rows = [[textwrap.wrap(c, WRAP) or [""] for c in r] for r in rows]     # long cells -> 2 lines
        w = [max(len(x) for r in rows for x in r[i]) for i in range(n)]
        line = "  ".join("-" * x for x in w)
        out.append(line)
        for k, r in enumerate(rows):
            for j in range(max(len(c) for c in r)):
                out.append("  ".join((c[j] if j < len(c) else "").ljust(w[i])
                                     for i, c in enumerate(r)).rstrip())
            if k == 0:
                out.append(line)
        out.append(line)
        block.clear()

    for l in lines:
        if l.startswith("|"):
            block.append(l)
            continue
        flush()
        t = l.replace("**", "").replace("`", "").rstrip()
        if t.startswith("# "):
            out += ["=" * 90, t[2:], "=" * 90]
        elif t.startswith("## "):
            out += [t[3:].upper(), "-" * len(t[3:])]
        else:
            out.append(t[:-2] if t.endswith("  ") else t)
    flush()
    return "\n".join(out)


def f(v, d=2, unit=""):
    return "-" if v is None else f"{v:.{d}f}{unit}"


def main(args):
    run_dir = run_folder(args.run)
    print(f"Run folder: {run_dir}")
    info = {k: load_json(run_dir / "models" / k / "training_info.json") for k in ("warm", "cold")}
    rows = {k: load_rows(run_dir / "logs" / "stage2" / k / "episodes.csv") for k in ("warm", "cold")}
    learn = {k: learning_stats(v) for k, v in rows.items()}
    stats_files = sorted((run_dir / "results").glob("stage2_statistics_*.json"))
    stats = load_json(stats_files[-1]) if stats_files else None
    res = (stats or {}).get("results", {})
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    png = out / "learning_curves_warm_vs_cold.png"
    has_png = plot(rows, png)

    status = {}
    w = info["warm"]
    d = (w or {}).get("transfer_max_action_diff")
    status["transfer"] = ("PASS" if d is not None and d < 1e-5 else ("NOT RUN" if not w else "CHECK"),
                          f"max action difference {d:.1e}" if d is not None else "-",
                          "warm PPO acts exactly like Stage 1 at start (< 1e-5)")
    for k in ("warm", "cold"):
        i = info[k]
        status[f"train_{k}"] = ("NOT RUN" if not i else ("PASS" if i.get("status") == "complete" else "CHECK"),
                                f"{i.get('timesteps_done', 0):,} steps, {i.get('episodes', 0)} episodes, "
                                f"last 50 success {f(i.get('last50_success_pct'), 0, '%')}" if i else "-",
                                "training finished")
    gw = res.get("warm", {}).get("gust_summary") or {}
    gs = res.get("stage1", {}).get("gust_summary") or {}
    gc = res.get("cold", {}).get("gust_summary") or {}
    status["wind_test"] = ("NOT RUN" if not gw else ("PASS" if gw.get("success_pct", 0) >= 80 else "CHECK"),
                           f"warm success {f(gw.get('success_pct'), 0, '%')}, RMSE {f(gw.get('rmse_m'), 3, ' m')}"
                           if gw else "-", "warm survives >= 80% of gust episodes (original target)")
    status["vs_stage1"] = ("NOT RUN" if not (gw and gs) else ("YES" if gw["rmse_m"] < gs["rmse_m"] else "NO"),
                           f"RMSE warm {f(gw.get('rmse_m'), 3)} vs Stage 1 {f(gs.get('rmse_m'), 3)} m"
                           if gw and gs else "-", "Stage 2 holds position better than Stage 1 in wind")
    lw, lc = learn["warm"], learn["cold"]
    status["vs_cold"] = ("NOT RUN" if not (lw and lc) else
                         ("YES" if lw["first50_success_pct"] > lc["first50_success_pct"] else "NO"),
                         f"first 50 episodes success warm {f((lw or {}).get('first50_success_pct'), 0, '%')} "
                         f"vs cold {f((lc or {}).get('first50_success_pct'), 0, '%')}" if lw and lc else "-",
                         "warm start learns faster than cold start")

    now = datetime.datetime.now()
    M = [f"# Stage 2 Summary: Hover in Wind (ArduPilot SITL + AirSim) - {run_dir.name.upper()}", "",
         f"**Run folder:** `{run_dir}`  ", f"**Generated:** {now:%Y-%m-%d %H:%M}  ",
         f"**Stage 1 network:** `{(w or info['cold'] or {}).get('stage1_run', '?')}`  ",
         "**Method:** PPO with wind gusts 0-20 m/s (new gust every 1-3 s); warm start from the Stage 1 network vs cold start "
         "from random weights, same settings otherwise", "",
         "## Executive summary", "", "| Check | Status | Key result | Rule |", "| --- | --- | --- | --- |"]
    for key, label in (("transfer", "Warm start loads Stage 1"), ("train_warm", "Training, warm start"),
                       ("train_cold", "Training, cold start"), ("wind_test", "Warm survives wind"),
                       ("vs_stage1", "Better than Stage 1 in wind"), ("vs_cold", "Warm learns faster than cold")):
        s, r, rule = status[key]
        M.append(f"| {label} | **{s}** | {r} | {rule} |")

    M += ["", "## Learning: warm vs cold start", "",
          "| Measure | Warm start | Cold start |", "| --- | --- | --- |"]
    for label, k, dd, u in (("Episodes", "episodes", 0, ""), ("Timesteps", "timesteps", 0, ""),
                            ("Success, first 50 episodes", "first50_success_pct", 0, "%"),
                            ("Mean distance, first 50 episodes", "first50_mean_dist_m", 2, " m"),
                            ("Success, last 50 episodes", "last50_success_pct", 0, "%"),
                            ("Mean distance, last 50 episodes", "last50_mean_dist_m", 2, " m"),
                            ("Steps until rolling success >= 80%", "steps_to_80pct", 0, "")):
        cell = lambda L: "-" if not L else ("not reached" if L.get(k) is None else
                                            (f"{L[k]:,}" if isinstance(L[k], int) else f(L[k], dd, u)))
        M.append(f"| {label} | {cell(lw)} | {cell(lc)} |")
    if has_png:
        M += ["", f"Learning curves: `{png}`"]

    if res:
        names = stats.get("names", {})
        M += ["", f"## In wind: all controllers, identical conditions (`{stats_files[-1].name}`)", "",
              "Gusts 0-20 m/s:", "", "| Controller | Success | RMSE (m) | Max (m) | Held within 0.5 m |",
              "| --- | --- | --- | --- | --- |"]
        for c in ("pid", "stage1", "warm", "cold"):
            s = (res.get(c) or {}).get("gust_summary")
            if s:
                M.append(f"| {names.get(c, c)} | {s['success_pct']:.0f}% | {s['rmse_m']:.3f} | "
                         f"{s['max_dist_m']:.3f} | {s['hold_pct']:.0f}% |")
        levels = stats.get("levels_mps", [])
        M += ["", "Constant wind, RMSE (m) / success:", "",
              "| Controller | " + " | ".join(f"{l:.0f} m/s" for l in levels) + " |",
              "| --- | " + " | ".join("---" for _ in levels) + " |"]
        for c in ("pid", "stage1", "warm", "cold"):
            ls = (res.get(c) or {}).get("level_summary")
            if ls:
                M.append(f"| {names.get(c, c)} | " + " | ".join(
                    f"{ls[str(l)]['rmse_m']:.3f} / {ls[str(l)]['success_pct']:.0f}%" if ls.get(str(l)) else "-"
                    for l in levels) + " |")
        if gw and gs:
            M += ["", f"Stage 2 (warm) vs Stage 1 in gusts: RMSE {gw['rmse_m']:.3f} vs {gs['rmse_m']:.3f} m "
                      f"({100 * (gs['rmse_m'] - gw['rmse_m']) / gs['rmse_m']:+.1f}%)."]
        if gw and gc:
            M += [f"Warm vs cold in gusts: RMSE {gw['rmse_m']:.3f} vs {gc['rmse_m']:.3f} m "
                  f"({100 * (gc['rmse_m'] - gw['rmse_m']) / gc['rmse_m']:+.1f}%)."]

    M += ["", "## Settings", ""]
    if w or info["cold"]:
        i = w or info["cold"]
        M += [f"- PPO: {json.dumps(i.get('ppo', {}))}", f"- Environment: {json.dumps(i.get('env', {}))}"]
    path = out / f"STAGE2_SUMMARY_{run_dir.name}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(M) + "\n")
    print(console_text(M))
    print(f"\nFull summary saved (Markdown, open in VS Code preview): {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    main(ap.parse_args())
