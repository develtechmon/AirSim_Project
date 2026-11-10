"""
STAGE 1 - RMSE BASELINE COMPARISON (ArduPilot)
==============================================
Same job as your AirSim rmse_baseline.py: per-episode RMSE table, PID vs BC,
and transfer efficiency.

Reads the newest runs/run<N>/results/statistics_summary_<date>.pkl
(latest run, or --run N, or --file).
Saves runs/run<N>/results/rmse_baseline_<same date>.txt

Run: python rmse_baseline.py
"""

import argparse
import pickle
from pathlib import Path

from train_imitation import run_folder

import numpy as np


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
    b, p, c = data["baseline"], data["policy"], data["comparison"]

    L = ["=" * 82, "STAGE 1 - RMSE BASELINE COMPARISON",
         f"PID Expert vs BC Policy | sensor mode: {data['sensor_mode']} | source: {path.name}",
         "=" * 82, "", "TABLE 1 - AGGREGATE METRICS", "-" * 72,
         f"{'Metric':<38}{'PID Expert':>12}{'BC Policy':>12}{'Gap':>10}", "-" * 72]
    for label, k in [("Position error RMSE (m)", "pos_rmse_mean"), ("Altitude RMSE (m)", "alt_rmse_mean"),
                     ("Altitude std (m)", "alt_std_mean"), ("Mean horiz. distance (m)", "dist_mean"),
                     ("Max horiz. distance (m)", "dist_max_mean"), ("End error (m)", "end_err_mean")]:
        L.append(f"  {label:<36}{b[k]:>12.4f}{p[k]:>12.4f}{p[k]-b[k]:>+10.4f}")
    L += ["-" * 72,
          f"  {'Position RMSE ratio (BC / PID)':<36}{'':>12}{c['pos_rmse_ratio']:>12.4f}",
          f"  {'Transfer efficiency, position (%)':<36}{'':>12}{c['transfer_efficiency_pos']:>12.2f}",
          f"  {'Altitude RMSE ratio (BC / PID)':<36}{'':>12}{c['alt_rmse_ratio']:>12.4f}",
          f"  {'Transfer efficiency, altitude (%)':<36}{'':>12}{c['transfer_efficiency_alt']:>12.2f}",
          "", "TABLE 2 - PER-EPISODE RMSE", "-" * 82,
          f"{'Ep':<4}{'Target alt':>11}{'PID pos':>10}{'BC pos':>10}{'PID alt':>10}{'BC alt':>10}"
          f"{'PID std':>10}{'BC std':>10}", "-" * 82]
    n = min(b["n_episodes"], p["n_episodes"])
    for i in range(n):
        L.append(f"{i+1:<4}{b['target_alts'][i]:>11.2f}{b['ep_pos_rmse'][i]:>10.4f}{p['ep_pos_rmse'][i]:>10.4f}"
                 f"{b['ep_alt_rmse'][i]:>10.4f}{p['ep_alt_rmse'][i]:>10.4f}"
                 f"{b['ep_alt_std'][i]:>10.4f}{p['ep_alt_std'][i]:>10.4f}")
    L += ["-" * 82,
          f"{'Mean':<4}{'':>11}{np.mean(b['ep_pos_rmse'][:n]):>10.4f}{np.mean(p['ep_pos_rmse'][:n]):>10.4f}"
          f"{np.mean(b['ep_alt_rmse'][:n]):>10.4f}{np.mean(p['ep_alt_rmse'][:n]):>10.4f}"
          f"{np.mean(b['ep_alt_std'][:n]):>10.4f}{np.mean(p['ep_alt_std'][:n]):>10.4f}",
          "", "Transfer efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%",
          f"Target: >= 90% (same target as the AirSim Stage 1 paper)",
          f"Result: position {c['transfer_efficiency_pos']:.1f}% | altitude {c['transfer_efficiency_alt']:.1f}%",
          "=" * 82]
    text = "\n".join(L)
    print(text)
    out = path.parent / path.name.replace("statistics_summary_", "rmse_baseline_").replace(".pkl", ".txt")
    with open(out, "w") as f:
        f.write(text)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="statistics_summary_<date>.pkl (default: newest in the run)")
    ap.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    main(ap.parse_args())
