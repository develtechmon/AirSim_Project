r"""
ANALYZE STAGE 2 STATISTICS (paired tests + gust recovery)  - no simulator needed
================================================================================

1. WHAT IT DOES
---------------
Reads the .pkl files written by collect_stage2_statistics.py and computes the numbers used
in the thesis discussion that the report table does not show:

    | Part | Question                                         | Method                          |
    |------|--------------------------------------------------|---------------------------------|
    | A    | Is PPO better in EVERY gust episode, or by luck? | paired per episode (same seed), |
    |      |                                                  | wins out of 10 + Wilcoxon p     |
    | B    | HOW does PPO win: smaller push or faster return? | pushes past 0.5 m, time back    |
    |      |                                                  | below 0.25 m, biggest push      |
    | C    | Constant wind: any difference?                   | paired, 8 episodes              |
    | D    | Repeat session (optional): reproducible?         | PPO session 1 vs session 2      |

    Wilcoxon signed-rank, exact two-sided p (computed here, no scipy needed).
    With 10 pairs the smallest possible p is 0.002 (= wins 10/10).


2. HOW TO RUN (in Stage_2\)
---------------------------
    py .\analyze_stage2_statistics.py

    Defaults: MAIN = newest stage2_statistics_*.pkl that contains warm + at least one other
              controller; REPEAT = newest .pkl that contains warm only (skipped if none).
    Other files:

    py .\analyze_stage2_statistics.py --main stage2_statistics_20261022_161620.pkl --repeat stage2_statistics_20261022_164145.pkl


3. EXPECTED RESULT (your run1, 22 Oct 2026)
-------------------------------------------
    | Check                                   | Your result                         |
    |-----------------------------------------|-------------------------------------|
    | A: PPO vs stock ArduPilot, gust RMSE    | wins 10/10, p = 0.002               |
    | B: pushes past 0.5 m (PPO / ArduPilot)  | 9 / 16, back below 0.25 m 1.5 / 2.6 s|
    | B: biggest push per episode             | no clear difference (p = 0.06)      |
    | C: PPO vs ArduPilot, constant wind      | 5/8, p = 0.74  (no difference)      |
    | D: repeat session, PPO RMSE             | 0.198 -> 0.226 m, still wins 10/10  |


4. OUTPUT FILES
---------------
    runs\run<N>\results\statistics\stage2_analysis_<date>.txt    printed tables
    runs\run<N>\results\statistics\stage2_analysis_<date>.json   all numbers
"""

import argparse
import datetime
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from drone_hover_disturbance_env import run_folder

NAMES = {"pid": "PID teacher", "stage1": "Stage 1 (BC)", "warm": "PPO warm", "cold": "PPO cold",
         "ardupilot": "Stock ArduPilot"}
HZ = 20.0
PUSH_M = 0.5          # a "push" = distance goes above this
BACK_M = 0.25         # ... and counts as recovered when back below this


def wilcoxon_exact(a, b):
    """Two-sided exact Wilcoxon signed-rank p for paired samples (zero differences dropped)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    absd = np.abs(d)[order]
    i = 0
    while i < n:                                   # average ranks for ties
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    w_obs = ranks[d > 0].sum()
    mean = ranks.sum() / 2
    dev = abs(w_obs - mean)
    hits = 0
    for signs in itertools.product((0, 1), repeat=n):
        if abs(np.dot(signs, ranks) - mean) >= dev - 1e-9:
            hits += 1
    return hits / 2 ** n


def pushes(trace):
    """List of (seconds from > PUSH_M back to < BACK_M) for every push in one episode."""
    tr = np.asarray(trace)
    out, i = [], 0
    while i < len(tr):
        if tr[i] > PUSH_M:
            j = i
            while j < len(tr) and tr[j] > BACK_M:
                j += 1
            if j < len(tr):
                out.append((j - i) / HZ)
            i = j
        i += 1
    return out


def paired(main, key, ctrl, other, eps_of, higher_better=False):
    a = np.array([e[key] for e in eps_of(main[ctrl])])
    b = np.array([e[key] for e in eps_of(main[other])])
    wins = int(np.sum(a > b) if higher_better else np.sum(a < b))
    return {"ppo": float(a.mean()), "other": float(b.mean()), "wins": wins, "n": len(a),
            "p": wilcoxon_exact(a, b)}


def pick(files, want_all):
    for f in reversed(files):
        res = pickle.load(open(f, "rb"))["results"]
        if "warm" in res and ((len(res) > 1) == want_all):
            return f
    return None


def main(args):
    run_dir = run_folder(args.run)
    sdir = run_dir / "results" / "statistics"
    files = sorted(sdir.glob("stage2_statistics_*.pkl"))
    mf = sdir / args.main if args.main else pick(files, True)
    rf = (sdir / args.repeat if args.repeat else pick(files, False))
    if mf is None:
        raise SystemExit(f"No stage2_statistics_*.pkl with warm + another controller in {sdir}")
    main_res = pickle.load(open(mf, "rb"))["results"]
    others = [c for c in ("ardupilot", "stage1", "pid", "cold") if c in main_res]
    gust = lambda r: r["gust"]
    const = lambda r: [e for v in r["levels"].values() for e in v]
    out = {"main_file": Path(mf).name, "repeat_file": Path(rf).name if rf else None, "A": {}, "B": {}, "C": {}}
    L = ["=" * 92, "STAGE 2 ANALYSIS - PAIRED TESTS AND GUST RECOVERY", f"Main file: {Path(mf).name}",
         "=" * 92, "", "A. GUSTS - PPO warm vs each controller, paired per episode (same seed)", "-" * 92,
         f"{'vs':<17}{'RMSE PPO/other':>18}{'wins':>7}{'p':>8}{'Dist PPO/other':>18}{'wins':>7}"
         f"{'Held% PPO/other':>17}{'wins':>6}", "-" * 92]
    for c in others:
        r = paired(main_res, "ep_rmse", "warm", c, gust)
        d = paired(main_res, "ep_mean_hdist", "warm", c, gust)
        h = paired(main_res, "ep_hold_pct", "warm", c, gust, higher_better=True)
        out["A"][c] = {"rmse": r, "dist": d, "hold": h}
        L.append(f"{NAMES[c]:<17}{r['ppo']:>8.3f}/{r['other']:<9.3f}{r['wins']:>4}/{r['n']}{r['p']:>8.3f}"
                 f"{d['ppo']:>8.3f}/{d['other']:<9.3f}{d['wins']:>4}/{d['n']}"
                 f"{h['ppo']:>8.1f}/{h['other']:<8.1f}{h['wins']:>3}/{h['n']}")
    L += ["", f"B. GUSTS - HOW: pushes past {PUSH_M} m, time back below {BACK_M} m, biggest push", "-" * 92,
          f"{'Controller':<17}{'pushes':>8}{'back in (mean s)':>18}{'biggest push (mean m)':>23}"
          f"{'vs PPO wins':>13}{'p':>8}", "-" * 92]
    for c in ["warm"] + others:
        st = [x for e in main_res[c]["gust"] for x in pushes(e["ep_dist_trace"])]
        mx = float(np.mean([e["ep_max_hdist"] for e in main_res[c]["gust"]]))
        row = {"pushes": len(st), "back_s": float(np.mean(st)) if st else None, "max_push_m": mx}
        tail = ""
        if c != "warm":
            m = paired(main_res, "ep_max_hdist", "warm", c, gust)
            row["max_push_vs_ppo"] = m
            tail = f"{m['wins']:>9}/{m['n']}{m['p']:>9.3f}"
        out["B"][c] = row
        back = f"{row['back_s']:.2f}" if st else "-"
        L.append(f"{NAMES[c]:<17}{len(st):>8}{back:>18}{mx:>23.3f}{tail}")
    L += ["", "C. CONSTANT WIND 5/10/15/20 - PPO vs each, paired (RMSE)", "-" * 92]
    for c in others:
        r = paired(main_res, "ep_rmse", "warm", c, const)
        out["C"][c] = r
        L.append(f"  vs {NAMES[c]:<17} RMSE {r['ppo']:.3f} vs {r['other']:.3f} m | wins {r['wins']}/{r['n']} | p {r['p']:.3f}")
    if rf:
        rep = pickle.load(open(rf, "rb"))["results"]["warm"]["gust"]
        s1 = np.array([e["ep_rmse"] for e in main_res["warm"]["gust"]])
        s2 = np.array([e["ep_rmse"] for e in rep])
        both = np.concatenate([s1, s2])
        out["D"] = {"session1_rmse": float(s1.mean()), "session2_rmse": float(s2.mean()),
                    "pooled_mean": float(both.mean()), "pooled_sd": float(both.std(ddof=1)), "n": len(both),
                    "per_episode_change": [float(x) for x in s2 - s1],
                    "vs": {c: {"wins": int(np.sum(s2 < np.array([e["ep_rmse"] for e in main_res[c]["gust"]]))),
                               "p": wilcoxon_exact(s2, [e["ep_rmse"] for e in main_res[c]["gust"]])}
                           for c in others}}
        D = out["D"]
        L += ["", f"D. REPEAT SESSION - PPO warm, gusts (repeat file: {Path(rf).name})", "-" * 92,
              f"  RMSE session 1 {D['session1_rmse']:.3f} m | session 2 {D['session2_rmse']:.3f} m | "
              f"pooled {D['pooled_mean']:.3f} +/- {D['pooled_sd']:.3f} m (n = {D['n']})",
              f"  per-episode change: {min(D['per_episode_change']):+.3f} .. {max(D['per_episode_change']):+.3f} m",
              "  session 2 PPO vs session 1 others: " + ", ".join(
                  f"{NAMES[c]} {v['wins']}/10 (p {v['p']:.3f})" for c, v in D["vs"].items())]
    L += ["", "=" * 92]
    text = "\n".join(L)
    print(text)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = sdir / f"stage2_analysis_{stamp}"
    with open(f"{base}.txt", "w") as f:
        f.write(text + "\n")
    json.dump(out, open(f"{base}.json", "w"), indent=2)
    print(f"Saved: {base}.txt / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=None, help="file name in results\\statistics\\ (default: newest with all)")
    ap.add_argument("--repeat", default=None, help="file name of a warm-only repeat (default: newest)")
    ap.add_argument("--run", default=None)
    main(ap.parse_args())
