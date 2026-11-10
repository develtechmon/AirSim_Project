r"""
STAGE 1 - COMMAND DISTRIBUTION, PID vs BC (ArduPilot)  - no simulator needed
============================================================================

1. WHAT IT DOES
---------------
Histogram of the velocity commands the PID expert and the BC network sent, on the SAME bins,
so the two strategies can be compared (did the network copy the teacher's behaviour?).

    | --axis       | Command          |
    |--------------|------------------|
    | vd (default) | vertical (down)  |
    | vn           | north            |
    | ve           | east             |


2. HOW TO RUN (in Stage_1\)
---------------------------
    py .\vd_distribution.py

    py .\vd_distribution.py --axis vn

    py .\vd_distribution.py --axis ve


3. EXPECTED RESULT (your run1)
------------------------------
    The two histograms overlap almost completely, centred on 0.
    vd std: PID 0.091 vs BC 0.091 m/s | vn std: 0.214 vs 0.196 | ve std: 0.180 vs 0.171
    (BC slightly smoother than the teacher - normal for a network that averages examples.)


4. OUTPUT FILES (folder made automatically)
-------------------------------------------
    runs\run<N>\results\vd_distribution\<axis>_distribution_<same date>.png
"""

import argparse
import pickle
from pathlib import Path

from train_imitation import run_folder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXES = {"vn": (0, "north"), "ve": (1, "east"), "vd": (2, "down")}


def main(args):
    if args.file:
        path = Path(args.file)
    else:
        run_dir = run_folder(args.run)
        files = sorted((run_dir / "results" / "statistics").glob("statistics_summary_*.pkl")) or \
            sorted((run_dir / "results").glob("statistics_summary_*.pkl"))      # old layout
        if not files:
            raise SystemExit(f"No statistics_summary in {run_dir / 'results' / 'statistics'} - "
                             "run collect_stage1_statistics.py first")
        path = files[-1]
    results = path.parent.parent if path.parent.name == "statistics" else path.parent
    print(f"Results folder: {results} | file: {path.name}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    col, name = AXES[args.axis]
    pid = np.array([a[col] for e in data["episodes"]["pid"] for a in e["actions"]])
    bc = np.array([a[col] for e in data["episodes"]["bc"] for a in e["actions"]])

    print(f"\n=== {args.axis.upper()} COMMAND DISTRIBUTION ({name}, m/s) ===")
    for label, v in (("PID", pid), ("BC ", bc)):
        print(f"{label}: n {len(v)} | mean {v.mean():+.4f} | std {v.std():.4f} | "
              f"median {np.median(v):+.4f} | min {v.min():+.4f} | max {v.max():+.4f}")

    edges = np.histogram_bin_edges(np.concatenate([pid, bc]), bins=40)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pid, bins=edges, alpha=0.55, label=f"PID expert (mean {pid.mean():+.3f})")
    ax.hist(bc, bins=edges, alpha=0.55, label=f"BC policy (mean {bc.mean():+.3f})")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{args.axis} command (m/s)")
    ax.set_ylabel("Steps")
    ax.set_title(f"Stage 1 {args.axis} command distribution, PID vs BC "
                 f"({data['sensor_mode']})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    (results / "vd_distribution").mkdir(parents=True, exist_ok=True)
    out = results / "vd_distribution" / path.name.replace("statistics_summary_", f"{args.axis}_distribution_") \
        .replace(".pkl", ".png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="statistics_summary_<date>.pkl (default: newest in the run)")
    ap.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    ap.add_argument("--axis", default="vd", choices=list(AXES))
    main(ap.parse_args())
