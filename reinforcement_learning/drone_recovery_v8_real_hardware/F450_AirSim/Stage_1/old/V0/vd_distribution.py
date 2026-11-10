"""
STAGE 1 - VERTICAL COMMAND DISTRIBUTION (ArduPilot)
===================================================
Same job as your AirSim vd_distribution.py, but PID and BC are now drawn
on the SAME bins, so the two strategies can actually be compared.

Reads the newest runs/run<N>/results/statistics_summary_<date>.pkl
(latest run, or --run N, or --file).
Saves runs/run<N>/results/vd_distribution_<same date>.png

Run: python vd_distribution.py            (vd = vertical command)
     python vd_distribution.py --axis vn  (north)  |  --axis ve (east)
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
        files = sorted((run_dir / "results").glob("statistics_summary_*.pkl"))
        if not files:
            raise SystemExit(f"No statistics_summary in {run_dir / 'results'} - "
                             "run collect_stage1_statistics.py first")
        path = files[-1]
    print(f"Run folder: {path.parent.parent} | file: {path.name}")
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
    out = path.parent / path.name.replace("statistics_summary_", f"{args.axis}_distribution_") \
        .replace(".pkl", ".png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="statistics_summary_<date>.pkl (default: newest in the run)")
    ap.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    ap.add_argument("--axis", default="vd", choices=list(AXES))
    main(ap.parse_args())
