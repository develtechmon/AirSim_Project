"""
MAKE STAGE 1 SUMMARY (ArduPilot)
================================
Builds the Stage 1 executive summary AUTOMATICALLY from the saved result files,
so no number is ever copied by hand. Run it again after any rerun.

Reads ONLY from one run folder (latest, or --run N):
    runs/run<N>/models/model_info.json                    training numbers (+ which dataset)
    runs/run<N>/demonstrations/<that dataset>.json        collection numbers
    runs/run<N>/logs/stage1/training_<date>.csv           loss per epoch
    runs/run<N>/results/test_hover_<newest>.json / .csv   test of the network alone
    runs/run<N>/results/statistics_summary_<newest>.json  PID vs BC comparison

Writes:
    runs/run<N>/results/STAGE1_SUMMARY_run<N>.md   (rewritten each time you run it)

Run: python make_stage1_summary.py            (latest run)
     python make_stage1_summary.py --run 1    (pick run 1)
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

from train_imitation import run_folder


def newest(folder, pattern):
    files = sorted(Path(folder).glob(pattern))
    return files[-1] if files else None


def load(path):
    return json.load(open(path)) if path and Path(path).exists() else None


def f(v, d=3):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def main(args):
    run_dir = run_folder(args.run)
    print(f"Run folder: {run_dir}")
    info = load(run_dir / "models" / "model_info.json")
    coll_path = Path(info["dataset"]).with_suffix(".json") if info and info.get("dataset") else \
        newest(run_dir / "demonstrations", "expert_demonstrations_*.json")
    if coll_path and not Path(coll_path).exists():          # files moved into runs/run<N>/
        coll_path = run_dir / "demonstrations" / Path(str(coll_path).replace("\\", "/")).name
    coll = load(coll_path)
    if info and info.get("training_log") and not Path(info["training_log"]).exists():
        info["training_log"] = str(run_dir / "logs" / "stage1" /
                                   Path(info["training_log"].replace("\\", "/")).name)
    test_path = newest(run_dir / "results", "test_hover_*.json")
    test = load(test_path)
    stats_path = newest(run_dir / "results", "statistics_summary_*.json")
    stats = load(stats_path)

    epochs = []
    if info and info.get("training_log") and Path(info["training_log"]).exists():
        with open(info["training_log"]) as fh:
            epochs = [(int(r["epoch"]), float(r["train_loss"]), float(r["val_loss"]))
                      for r in csv.DictReader(fh)]
    test_rows = []
    if test_path and test_path.with_suffix(".csv").exists():
        with open(test_path.with_suffix(".csv")) as fh:
            test_rows = list(csv.DictReader(fh))

    # ---------------- pass rules ----------------
    status = {}
    if coll:
        hz = coll["settings"]["hz"]
        dt = coll["timing"]["dt_mean"]
        over_pct = 100.0 * coll["timing"]["overruns"] / max(1, coll["samples"])
        ok = abs(dt - 1 / hz) <= 0.05 / hz and over_pct <= 1.0 and coll["settled_pct"] >= 80
        status["collect"] = ("PASS" if ok else "CHECK",
                             f"{coll['samples']:,} samples, loop {dt*1000:.1f} ms, "
                             f"expert settled {coll['settled_pct']:.0f}%",
                             "loop within 5% of target, overruns <= 1%, expert settled >= 80%")
    else:
        status["collect"] = ("NOT RUN", "-", "-")
    if info:
        d = info.get("transfer_max_action_diff")
        status["train"] = ("PASS" if d is not None and d < 1e-5 else "CHECK",
                           f"best val loss {info['best_val_loss']:.4f} (epoch {info['best_epoch']}), "
                           f"transfer diff {f(d, 1) if d is None else f'{d:.1e}'}",
                           "transfer to PPO identical (diff < 1e-5)")
    else:
        status["train"] = ("NOT RUN", "-", "-")
    if test:
        status["test"] = ("PASS" if test["success_rate_pct"] >= 90 else "CHECK",
                          f"success {test['success_rate_pct']:.0f}% ({test['episodes']} episodes), "
                          f"mean end error {test['end_err_mean_m']:.2f} m",
                          "success >= 90%")
    else:
        status["test"] = ("NOT RUN", "-", "-")
    if stats:
        te = stats["comparison"]["transfer_efficiency_pos"]
        status["stats"] = ("PASS" if te >= 90 else "CHECK",
                           f"transfer efficiency {te:.1f}% (position), "
                           f"{stats['comparison']['transfer_efficiency_alt']:.1f}% (altitude)",
                           "transfer efficiency >= 90%")
    else:
        status["stats"] = ("NOT RUN", "-", "-")

    # ---------------- markdown ----------------
    now = datetime.datetime.now()
    mode = (coll or {}).get("sensor_mode", "?")
    M = [f"# Stage 1 Summary: Hover Training (ArduPilot SITL + AirSim) - {run_dir.name.upper()}", "",
         f"**Run folder:** `{run_dir}`  ",
         f"**Generated:** {now:%Y-%m-%d %H:%M}  ",
         f"**Sensor mode:** {mode}  ",
         f"**Firmware:** {(coll or {}).get('firmware', '?')}  ",
         f"**Method:** Behavioural cloning of the expert ({(coll or {}).get('expert', 'cascade PID, first run')})  ",
         "**Observation:** 13 (position minus target, velocity, quaternion, body rates)  ",
         "**Action:** 3 velocity commands (vx, vy, vz), GUIDED mode", "",
         "All numbers below are read from the saved result files. Each status follows the rule "
         "in its row.", "",
         "## Executive summary", "",
         "| Step | Script | Status | Key result | Pass rule |",
         "| --- | --- | --- | --- | --- |"]
    for step, script, key in [("1. Data collection", "collect_demonstration.py", "collect"),
                              ("2. Training", "train_imitation.py", "train"),
                              ("3. Test (network alone)", "test_hover_policy.py", "test"),
                              ("4. PID vs BC", "collect_stage1_statistics.py", "stats")]:
        s, r, rule = status[key]
        M.append(f"| {step} | `{script}` | **{s}** | {r} | {rule} |")

    M += ["", "## Overall performance", "", "```"]
    if test:
        M.append(f"Test success rate:     {test['success_rate_pct']:.0f}%  "
                 f"(rule: {test.get('success_definition', '?')})")
        M.append(f"Test settled rate:     {test['settled_rate_pct']:.0f}%  "
                 f"(end error < {test.get('settled_threshold_m', '?')} m)")
        M.append(f"Test mean end error:   {test['end_err_mean_m']:.3f} m")
    if stats:
        M.append(f"Position RMSE PID/BC:  {stats['baseline']['pos_rmse_mean']:.3f} / "
                 f"{stats['policy']['pos_rmse_mean']:.3f} m")
        M.append(f"Transfer efficiency:   {stats['comparison']['transfer_efficiency_pos']:.1f}% "
                 f"(position), {stats['comparison']['transfer_efficiency_alt']:.1f}% (altitude)")
    if info:
        M.append(f"Best validation loss:  {info['best_val_loss']:.4f} (epoch {info['best_epoch']})")
    if coll:
        M.append(f"Dataset:               {coll['samples']:,} samples, "
                 f"{coll['episodes_done']} episodes")
    M.append("```")

    if coll:
        s = coll["settings"]
        M += ["", "## Step 1: Data collection (`collect_demonstration.py`)", "",
              f"Command: `py .\\collect_demonstration.py --episodes {s['episodes']} --steps {s['steps']}`", "",
              "| Item | Value |", "| --- | --- |",
              f"| File | `{Path(coll_path).with_suffix('.pkl').name}` |",
              f"| Samples / episodes | {coll['samples']:,} / {coll['episodes_done']} |",
              f"| Control rate | {s['hz']:.0f} Hz target, {coll['timing']['dt_mean']*1000:.1f} ms measured "
              f"(max {coll['timing']['dt_max']*1000:.1f} ms, {coll['timing']['overruns']} overruns) |",
              f"| Heights | {s['alt_min']}-{s['alt_max']} m, targets +/-{s['xy_offset']} m xy, "
              f"+/-{s['z_offset']} m z |",
              f"| PID expert settled | {coll['settled_pct']:.0f}% (end error < {coll['settled_threshold_m']} m), "
              f"mean end error {coll['end_err_mean_m']:.3f} m |",
              f"| Safety stop | {'yes' if coll['aborted'] else 'none'} |"]

    if info:
        M += ["", "## Step 2: Training (`train_imitation.py`)", "",
              "| Item | Value |", "| --- | --- |",
              f"| Architecture | {info['architecture']}, {info['params']:,} parameters |",
              f"| Split | {info['split']}" + (f": {info['train_samples']:,} train / {info['val_samples']:,} validation"
                                               if 'train_samples' in info else "") + " |",
              f"| Settings | {info['epochs']} epochs, batch {info['batch_size']}, lr {info['lr']}, seed {info['seed']} |",
              f"| Best validation loss | {info['best_val_loss']:.4f} at epoch {info['best_epoch']} (saved model) |",
              f"| Validation RMSE per axis | vx {info['val_rmse_per_axis_mps'][0]:.3f}, "
              f"vy {info['val_rmse_per_axis_mps'][1]:.3f}, vz {info['val_rmse_per_axis_mps'][2]:.3f} m/s |",
              f"| Transfer to PPO | max action difference {info['transfer_max_action_diff']} |",
              f"| Loss log | `{info['training_log']}` |"]
        if epochs:
            pick = sorted({1, 10, 25, 50, 100, info["best_epoch"], epochs[-1][0]})
            M += ["", "Loss at selected epochs:", "", "| Epoch | Train loss | Val loss |",
                  "| --- | --- | --- |"]
            for e, tl, vl in epochs:
                if e in pick:
                    tag = " (best)" if e == info["best_epoch"] else ""
                    M.append(f"| {e}{tag} | {tl:.4f} | {vl:.4f} |")

    if test:
        M += ["", "## Step 3: Test, network alone (`test_hover_policy.py`)", "",
              f"Rule: success = {test.get('success_definition', '?')}; settled = end error < "
              f"{test.get('settled_threshold_m', '?')} m.", "",
              "| Episode | Start error (m) | End error (m) | RMSE (m) | Max error (m) | Result |",
              "| --- | --- | --- | --- | --- | --- |"]
        for r in test_rows:
            res = "success" if r["success"] == "1" else f"fail ({r['reason']})"
            M.append(f"| {r['episode']} | {r['start_err_m']} | {r['end_err_m']} | {r['rmse_m']} | "
                     f"{r['max_err_m']} | {res} |")

    if stats:
        b, p, c = stats["baseline"], stats["policy"], stats["comparison"]
        M += ["", "## Step 4: PID vs BC (`collect_stage1_statistics.py`)", "",
              f"Same {b['n_episodes']} targets for both (seed 42). Source: `{stats_path.name}`", "",
              "| Metric | PID expert | BC policy | Gap |", "| --- | --- | --- | --- |"]
        for label, k in [("Position error RMSE (m)", "pos_rmse_mean"), ("Altitude RMSE (m)", "alt_rmse_mean"),
                         ("Max horizontal distance (m)", "dist_max_mean"), ("End error (m)", "end_err_mean"),
                         ("Success rate (%)", "success_rate"), ("Settled rate (%)", "settled_rate")]:
            M.append(f"| {label} | {b[k]:.3f} | {p[k]:.3f} | {p[k]-b[k]:+.3f} |")
        M += [f"| Transfer efficiency, position | | {c['transfer_efficiency_pos']:.1f}% | |",
              f"| Transfer efficiency, altitude | | {c['transfer_efficiency_alt']:.1f}% | |", "",
              "Transfer efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%."]

    M += ["", "## Target vs achieved", "",
          "| Target | Rule | Achieved | Status |", "| --- | --- | --- | --- |"]
    if info:
        M.append(f"| Transfer to Stage 2 works | action diff < 1e-5 | "
                 f"{info['transfer_max_action_diff']} | {status['train'][0]} |")
    if test:
        M.append(f"| Network hovers alone | success >= 90% | {test['success_rate_pct']:.0f}% | "
                 f"{status['test'][0]} |")
    if stats:
        M.append(f"| Network matches the PID | transfer efficiency >= 90% | "
                 f"{stats['comparison']['transfer_efficiency_pos']:.1f}% | {status['stats'][0]} |")
    if coll:
        M.append(f"| Fixed control rate | within 5% of {1000/coll['settings']['hz']:.0f} ms | "
                 f"{coll['timing']['dt_mean']*1000:.1f} ms | {status['collect'][0]} |")

    M += ["", "## Files", "", "| File | Use |", "| --- | --- |",
          f"| `{Path(coll_path).with_suffix('.pkl') if coll else '-'}` | training data |",
          f"| `{run_dir / 'models' / 'hover_policy_best.pth'}` | trained network |",
          f"| `{run_dir / 'models' / 'obs_norm.npz'}` | input scaling (Stage 2 reuses it) |",
          f"| `{run_dir / 'models' / 'stage1_ppo_init.zip'}` | Stage 2 starting point |",
          f"| `{info['training_log'] if info else '-'}` | loss curve data (Graph 1) |",
          f"| `{test_path if test else '-'}` | test success rate |",
          f"| `{stats_path if stats else '-'}` | PID vs BC numbers (Graphs 2-5, RMSE table) |"]

    ready = [("Model trained (`hover_policy_best.pth`)", (run_dir / "models" / "hover_policy_best.pth").exists()),
             ("Stage 2 start file (`stage1_ppo_init.zip`)", (run_dir / "models" / "stage1_ppo_init.zip").exists()),
             ("Transfer to PPO verified", status["train"][0] == "PASS"),
             ("Test success >= 90%", status["test"][0] == "PASS"),
             ("Transfer efficiency >= 90%", status["stats"][0] == "PASS"),
             ("Data collected in optical-flow mode", mode == "optflow")]
    M += ["", "## Readiness for Stage 2", ""]
    M += [f"- [{'x' if ok else ' '}] {t}" for t, ok in ready]

    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"STAGE1_SUMMARY_{run_dir.name}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(M) + "\n")
    print("\n".join(M[:30]))
    print(f"\n... full summary saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    main(ap.parse_args())
