"""
STAGE 1 STATISTICS COLLECTION FOR AIRSIM
==========================================
Produces statistics_summary.pkl with IDENTICAL structure to your PX4 pkl.
Your existing rmse_baseline.py and vd_distribution.py work unchanged.

Usage:
    python collect_stage1_statistics.py

Output: ./results/statistics_summary.pkl
        ./results/statistics_report.txt

Prerequisites:
    - AirSim open and running
    - hover_policy_best.pth trained and saved in ./models/
    - pid_expert.py in same directory
"""

import airsim
import numpy as np
import torch
import torch.nn as nn
import time
import pickle
import os
from pathlib import Path
from pid_expert import PIDExpertHover


# ── Must match train_imitation_v2.py exactly ──────────────────────────────────
class HoverPolicy(nn.Module):
    def __init__(self, state_dim=13, action_dim=3):
        super(HoverPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def predict(self, obs):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            return self.network(t).squeeze(0).numpy()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_state(client):
    s   = client.getMultirotorState()
    pos = s.kinematics_estimated.position
    vel = s.kinematics_estimated.linear_velocity
    ori = s.kinematics_estimated.orientation
    ang = s.kinematics_estimated.angular_velocity
    obs = np.array([
        pos.x_val, pos.y_val, pos.z_val,
        vel.x_val, vel.y_val, vel.z_val,
        ori.w_val, ori.x_val, ori.y_val, ori.z_val,
        ang.x_val, ang.y_val, ang.z_val
    ], dtype=np.float32)
    state_dict = {
        'position':         np.array([pos.x_val, pos.y_val, pos.z_val]),
        'velocity':         np.array([vel.x_val, vel.y_val, vel.z_val]),
        'orientation':      np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val]),
        'angular_velocity': np.array([ang.x_val, ang.y_val, ang.z_val])
    }
    return obs, state_dict, pos, vel


def compute_histogram(values, n_bins=40):
    counts, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {
        'x':      centres.tolist(),
        'y':      counts.tolist(),
        'edges':  edges.tolist(),
        'width':  float(edges[1] - edges[0]),
        'mean':   float(np.mean(values)),
        'std':    float(np.std(values)),
        'median': float(np.median(values)),
    }


def run_episode(client, controller, controller_type,
                start_x, start_y, steps, target_altitude):
    """Run one episode, return altitude list, distance list, action list."""
    altitudes  = []
    distances  = []
    actions    = []

    for step in range(steps):
        obs, state_dict, pos, vel = get_state(client)

        if controller_type == 'PID':
            action = controller.get_action(state_dict)
        else:
            action = controller.predict(obs)

        client.moveByVelocityAsync(
            float(action[0]), float(action[1]), float(action[2]),
            duration=0.05,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()

        alt  = -pos.z_val
        dist = np.sqrt(pos.x_val**2 + pos.y_val**2)

        altitudes.append(float(alt))
        distances.append(float(dist))
        actions.append(action.tolist())

        # Safety stop
        if alt < 1.0 or alt > 80.0 or dist > 30.0:
            print(f"    ⚠️  Early stop at step {step} "
                  f"(alt={alt:.1f}m dist={dist:.1f}m)")
            # Pad remaining steps with last value
            while len(altitudes) < steps:
                altitudes.append(float(alt))
                distances.append(float(dist))
                actions.append(action.tolist())
            break

    return altitudes, distances, actions


def build_stats(label, episode_altitudes, episode_distances,
                all_actions, target_alt, starts):
    """Build stats dict matching PX4 pkl structure exactly."""
    n_ep    = len(episode_altitudes)
    ep_means, ep_stds, ep_errors, ep_rmse = [], [], [], []
    ep_drift_rates, ep_dist_means, ep_dist_max = [], [], []

    for i, alts in enumerate(episode_altitudes):
        arr   = np.array(alts)
        errs  = np.abs(arr - target_alt)
        dists = np.array(episode_distances[i])
        ep_means.append(float(np.mean(arr)))
        ep_stds.append(float(np.std(arr)))
        ep_errors.append(float(np.mean(errs)))
        ep_rmse.append(float(np.sqrt(np.mean(errs**2))))
        # drift rate: slope of distance over steps
        steps_arr = np.arange(len(dists))
        slope = float(np.polyfit(steps_arr, dists, 1)[0]) if len(dists) > 1 else 0.0
        ep_drift_rates.append(slope)
        ep_dist_means.append(float(np.mean(dists)))
        ep_dist_max.append(float(np.max(dists)))

    all_alts = [a for ep in episode_altitudes for a in ep]
    all_dists = [d for ep in episode_distances for d in ep]
    all_errs  = np.abs(np.array(all_alts) - target_alt)

    stats = {
        'label':           label,
        'n_episodes':      n_ep,
        'target_alt':      float(target_alt),
        'alt_mean':        float(np.mean(all_alts)),
        'alt_error_mean':  float(np.mean(all_errs)),
        'alt_error_std':   float(np.std(all_errs)),
        'alt_error_max':   float(np.max(all_errs)),
        'alt_rmse_mean':   float(np.mean(ep_rmse)),
        'alt_rmse_std':    float(np.std(ep_rmse)),
        'alt_std_mean':    float(np.mean(ep_stds)),
        'alt_std_std':     float(np.std(ep_stds)),
        'alt_global_std':  float(np.std(all_alts)),
        'drift_rate_mean': float(np.mean(ep_drift_rates)),
        'drift_rate_std':  float(np.std(ep_drift_rates)),
        'dist_mean':       float(np.mean(all_dists)),
        'dist_max_mean':   float(np.mean(ep_dist_max)),
        'dist_max_abs':    float(np.max(ep_dist_max)),
        'success_rate':    100.0,
        'ep_means':        ep_means,
        'ep_stds':         ep_stds,
        'ep_errors':       ep_errors,
        'ep_rmse':         ep_rmse,
        'ep_drift_rates':  ep_drift_rates,
        'ep_dist_means':   ep_dist_means,
        'ep_dist_max':     ep_dist_max,
        'all_altitudes':   all_alts,
        'episode_altitudes':  episode_altitudes,
        'episode_distances':  episode_distances,
    }

    # Action stats — only for policy, not baseline
    if all_actions:
        acts = np.array(all_actions)  # shape (N, 3)
        stats['action_mean']    = acts.mean(axis=0).tolist()
        stats['action_std']     = acts.std(axis=0).tolist()
        stats['action_max_abs'] = np.abs(acts).max(axis=0).tolist()
        stats['all_actions']    = acts.tolist()
        stats['action_histogram'] = {
            'vn': compute_histogram(acts[:, 0]),
            've': compute_histogram(acts[:, 1]),
            'vd': compute_histogram(acts[:, 2]),
        }

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    TARGET_ALT   = 30.0
    NUM_EPISODES = 10
    STEPS        = 500
    MODEL_PATH   = './models/hover_policy_best.pth'
    OUTPUT_DIR   = './results'

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STAGE 1 STATISTICS COLLECTION (AirSim)")
    print("="*70)
    print(f"Target altitude : {TARGET_ALT} m")
    print(f"Episodes        : {NUM_EPISODES} per controller")
    print(f"Steps/episode   : {STEPS}")
    print(f"Output          : {OUTPUT_DIR}/statistics_summary.pkl")
    print("="*70 + "\n")

    # Load BC policy
    if not os.path.exists(MODEL_PATH):
        print(f"❌ {MODEL_PATH} not found. Train first with train_imitation_v2.py")
        return
    bc_policy = HoverPolicy(state_dim=13, action_dim=3)
    bc_policy.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    bc_policy.eval()
    print(f"✅ BC policy loaded\n")

    pid_expert = PIDExpertHover(target_altitude=TARGET_ALT)

    # Random starts — same seed so PID and BC face identical conditions
    np.random.seed(42)
    starts = [(float(np.random.uniform(-2, 2)),
               float(np.random.uniform(-2, 2)))
              for _ in range(NUM_EPISODES)]

    # Connect
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected\n")

    # One-time init
    client.reset()
    time.sleep(1.0)
    client.enableApiControl(True)
    client.armDisarm(True)
    time.sleep(0.5)
    client.takeoffAsync().join()
    time.sleep(1.0)
    client.moveToPositionAsync(0, 0, -TARGET_ALT, 5, timeout_sec=25).join()
    time.sleep(1.5)
    print(f"✅ Drone at {TARGET_ALT}m. Starting collection...\n")

    # ── PART 1: PID Expert ────────────────────────────────────────────────────
    print("="*70)
    print("PART 1: PID Expert (baseline)")
    print("="*70)
    pid_ep_alts, pid_ep_dists = [], []

    for ep in range(NUM_EPISODES):
        ox, oy = starts[ep]
        print(f"  Episode {ep+1:2d}/{NUM_EPISODES}  "
              f"start=({ox:.1f},{oy:.1f})...", end=' ', flush=True)

        # Reposition
        pose = airsim.Pose(
            airsim.Vector3r(ox, oy, -TARGET_ALT),
            airsim.to_quaternion(0, 0, 0)
        )
        client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.4)
        client.enableApiControl(True)
        client.armDisarm(True)
        time.sleep(0.3)
        pid_expert.reset()

        alts, dists, _ = run_episode(
            client, pid_expert, 'PID',
            ox, oy, STEPS, TARGET_ALT)

        pid_ep_alts.append(alts)
        pid_ep_dists.append(dists)

        errs = np.abs(np.array(alts) - TARGET_ALT)
        rmse = np.sqrt(np.mean(errs**2))
        print(f"mean_alt={np.mean(alts):.3f}m  rmse={rmse:.4f}m  "
              f"max_drift={np.max(dists):.3f}m")

    # ── PART 2: BC Policy ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PART 2: BC Policy")
    print("="*70)
    bc_ep_alts, bc_ep_dists, bc_all_actions = [], [], []

    for ep in range(NUM_EPISODES):
        ox, oy = starts[ep]   # same starts as PID for fair comparison
        print(f"  Episode {ep+1:2d}/{NUM_EPISODES}  "
              f"start=({ox:.1f},{oy:.1f})...", end=' ', flush=True)

        pose = airsim.Pose(
            airsim.Vector3r(ox, oy, -TARGET_ALT),
            airsim.to_quaternion(0, 0, 0)
        )
        client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.4)
        client.enableApiControl(True)
        client.armDisarm(True)
        time.sleep(0.3)

        alts, dists, acts = run_episode(
            client, bc_policy, 'BC',
            ox, oy, STEPS, TARGET_ALT)

        bc_ep_alts.append(alts)
        bc_ep_dists.append(dists)
        bc_all_actions.extend(acts)

        errs = np.abs(np.array(alts) - TARGET_ALT)
        rmse = np.sqrt(np.mean(errs**2))
        print(f"mean_alt={np.mean(alts):.3f}m  rmse={rmse:.4f}m  "
              f"max_drift={np.max(dists):.3f}m")

    # ── Build stats dicts ─────────────────────────────────────────────────────
    baseline_stats = build_stats(
        'PID Expert (AirSim)', pid_ep_alts, pid_ep_dists,
        None, TARGET_ALT, starts)

    policy_stats = build_stats(
        'Imitation Policy (BC)', bc_ep_alts, bc_ep_dists,
        bc_all_actions, TARGET_ALT, starts)

    # Comparison
    comparison = {
        'alt_error_gap':   policy_stats['alt_error_mean'] - baseline_stats['alt_error_mean'],
        'alt_rmse_gap':    policy_stats['alt_rmse_mean']  - baseline_stats['alt_rmse_mean'],
        'alt_std_gap':     policy_stats['alt_std_mean']   - baseline_stats['alt_std_mean'],
        'dist_gap':        policy_stats['dist_mean']      - baseline_stats['dist_mean'],
        'drift_rate_gap':  policy_stats['drift_rate_mean']- baseline_stats['drift_rate_mean'],
        'rmse_ratio':      policy_stats['alt_rmse_mean']  / baseline_stats['alt_rmse_mean'],
        'std_ratio':       policy_stats['alt_std_mean']   / baseline_stats['alt_std_mean'],
        'transfer_efficiency': (1 - (policy_stats['alt_rmse_mean'] -
                                      baseline_stats['alt_rmse_mean']) /
                                      baseline_stats['alt_rmse_mean']) * 100,
    }

    # ── Save pkl ──────────────────────────────────────────────────────────────
    summary = {
        'baseline':    baseline_stats,
        'policy':      policy_stats,
        'comparison':  comparison,
        'target_alt':  TARGET_ALT,
        'num_episodes':NUM_EPISODES,
        'max_steps':   STEPS,
        'starts':      starts,
    }

    pkl_path = os.path.join(OUTPUT_DIR, 'statistics_summary.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(summary, f)
    print(f"\n✅ statistics_summary.pkl → {pkl_path}")

    # ── Save report ───────────────────────────────────────────────────────────
    bl = baseline_stats
    po = policy_stats
    cp = comparison

    lines = [
        "="*72,
        "STAGE 1 STATISTICS REPORT",
        "Behavioural Cloning: PID Expert vs BC Policy (AirSim SimpleFlight)",
        f"Target: {TARGET_ALT}m | Episodes: {NUM_EPISODES} | Steps: {STEPS}",
        "="*72,
        "",
        "TABLE 1 — MAIN COMPARISON",
        "-"*72,
        f"{'Metric':<38} {'PID Expert':>12} {'BC Policy':>12} {'Gap':>10}",
        "-"*72,
    ]
    metrics = [
        ("Mean altitude (m)",         'alt_mean'),
        ("Mean altitude error (m)",   'alt_error_mean'),
        ("RMSE from target (m)",      'alt_rmse_mean'),
        ("Altitude std (m)",          'alt_std_mean'),
        ("Mean horiz. drift (m)",     'dist_mean'),
        ("Max horiz. drift (m)",      'dist_max_mean'),
    ]
    for label, key in metrics:
        bv = bl[key]; pv = po[key]
        lines.append(f"  {label:<36} {bv:>12.4f} {pv:>12.4f} {pv-bv:>+10.4f}")

    lines += [
        "-"*72,
        f"  {'RMSE ratio (policy/baseline)':<36} {'':>12} {cp['rmse_ratio']:>12.4f}",
        f"  {'Transfer efficiency (%)':<36} {'':>12} {cp['transfer_efficiency']:>12.2f}",
        "",
        "TABLE 2 — PER-EPISODE RMSE",
        "-"*82,
        f"{'Ep':<4} {'BL alt':>8} {'BL std':>7} {'BL RMSE':>8} "
        f"{'BC alt':>8} {'BC std':>7} {'BC RMSE':>8}",
        "-"*82,
    ]
    for i in range(NUM_EPISODES):
        lines.append(
            f"{i+1:<4} {bl['ep_means'][i]:>8.3f} {bl['ep_stds'][i]:>7.4f} "
            f"{bl['ep_rmse'][i]:>8.4f} "
            f"{po['ep_means'][i]:>8.3f} {po['ep_stds'][i]:>7.4f} "
            f"{po['ep_rmse'][i]:>8.4f}")
    lines += [
        "-"*82,
        f"{'Mean':<4} {np.mean(bl['ep_means']):>8.3f} {np.mean(bl['ep_stds']):>7.4f} "
        f"{np.mean(bl['ep_rmse']):>8.4f} "
        f"{np.mean(po['ep_means']):>8.3f} {np.mean(po['ep_stds']):>7.4f} "
        f"{np.mean(po['ep_rmse']):>8.4f}",
        f"\nRMSE ratio: {cp['rmse_ratio']:.2f}×",
        f"Transfer efficiency: {cp['transfer_efficiency']:.2f}%",
        "",
        "ACTION DISTRIBUTION — BC Policy",
        "-"*72,
    ]
    for ch, label in [('vn','vn (north)'),('ve','ve (east)'),('vd','vd (down)')]:
        h = po['action_histogram'][ch]
        lines.append(f"  {label}: mean={h['mean']:+.4f}  std={h['std']:.4f}  "
                     f"median={h['median']:+.4f}")
    lines += ["", "="*72]

    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, 'statistics_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"✅ statistics_report.txt → {report_path}")
    print("\n" + report_text)

    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()
