"""
EXPORT STAGE 2 POLICY TO ONNX (for the Raspberry Pi Zero 2 on the real drone)
=============================================================================
Runs on the PC. Turns the trained network into ONE .onnx file the Pi can run
without PyTorch or Stable-Baselines3 (the Pi only needs onnxruntime + numpy).

Inside the .onnx file (so the Pi cannot get the scaling wrong):
    raw observation (13)  ->  (obs - mean) / std, clipped to +/-10   (Stage 1 obs_norm.npz)
                          ->  actor network (13-256-256-128-3)
                          ->  action clipped to +/-5                 (same as SITL)
    The Pi script then applies the same send limits as SITL (2.0 m/s xy, 1.0 m/s z).

Also stored in the file (metadata): the training heading, so the Pi can refuse to fly
when the drone faces a direction the network never saw (see real_fan_test.py).

Check: the ONNX output is compared with the original PPO model on 2000 random
observations. PASS = max difference < 1e-4 m/s.

Usage (in Stage_2):
    py .\\export_onnx.py                  (warm  -> runs\\run1\\onnx\\warm.onnx)
    py .\\export_onnx.py --model stage1   (Stage 1 network, for the before/after comparison)
"""

import argparse
import json
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from drone_hover_disturbance_env import ACT_LIMIT, CLIP_OBS, run_folder, stage1_folder


class OnnxActor(nn.Module):
    def __init__(self, policy, mean, std):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, obs_raw):
        x = torch.clamp((obs_raw - self.mean) / self.std, -CLIP_OBS, CLIP_OBS)
        return torch.clamp(self.action_net(self.policy_net(x)), -ACT_LIMIT, ACT_LIMIT)


def training_heading(mean, std):
    """Heading (deg) of the mean training quaternion, and how much it varied."""
    w, x, y, z = mean[6:10]
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return yaw, float(std[9])


def main(args):
    warnings.filterwarnings("ignore")          # torch export notices, not errors
    run_dir = run_folder(args.run)
    s1 = stage1_folder(args.stage1_run)
    if args.model == "warm":
        zip_path = run_dir / "models" / "warm" / "hover_disturbance_policy.zip"
    else:
        zip_path = s1 / "models" / "stage1_ppo_init.zip"
    norm_path = s1 / "models" / "obs_norm.npz"
    out_dir = run_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model}.onnx"

    print("\n" + "=" * 70)
    print("📦 EXPORT POLICY TO ONNX (for Raspberry Pi Zero 2)")
    print("=" * 70)
    print(f"[1/3] Loading model: {zip_path}")
    for p in (zip_path, norm_path):
        if not p.exists():
            raise SystemExit(f"   ❌ not found: {p}")
    model = PPO.load(str(zip_path), device="cpu")
    norm = np.load(norm_path)
    mean, std = norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
    yaw, qz_std = training_heading(mean, std)
    print(f"   ✅ Loaded. Input scaling: {norm_path}")
    print(f"   🧭 Training heading: {yaw:.0f} deg (quaternion z spread {qz_std:.3f})")

    print("\n[2/3] Exporting...")
    actor = OnnxActor(model.policy, mean, std).eval()
    dummy = torch.zeros(1, 13)
    torch.onnx.export(actor, dummy, str(out), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(str(out))
    meta = {"model": args.model, "source": str(zip_path), "obs_norm": str(norm_path),
            "training_heading_deg": round(yaw, 1), "quat_z_std": round(qz_std, 4),
            "act_limit": ACT_LIMIT, "clip_obs": CLIP_OBS,
            "obs": "pos-target NED(3), vel NED(3), quat w,x,y,z(4), rates p,q,r(3) - raw, unscaled",
            "action": "velocity N,E,D m/s (GUIDED)"}
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, json.dumps(v)
    onnx.save(m, str(out))
    print(f"   ✅ Saved: {out}")

    print("\n[3/3] Checking ONNX = original model...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    raw = (mean + std * rng.uniform(-3, 3, (2000, 13))).astype(np.float32)
    a_onnx = sess.run(None, {"obs": raw})[0]
    a_ppo = model.predict(np.clip((raw - mean) / std, -CLIP_OBS, CLIP_OBS), deterministic=True)[0]
    diff = float(np.abs(a_onnx - a_ppo).max())
    ok = diff < 1e-4
    print(f"   max |ONNX - PPO| action difference: {diff:.2e} m/s -> {'PASS' if ok else 'FAIL'}")
    print("\n" + "=" * 70)
    if not ok:
        print("❌ EXPORT FAILED - do not use this file")
        print("=" * 70 + "\n")
        raise SystemExit(1)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)
    print(f"Copy to the Pi:  {out}")
    print("Next: see the Stage 2 guide, section 'Real drone'.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="warm", choices=["warm", "stage1"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--stage1-run", default=None)
    main(ap.parse_args())
