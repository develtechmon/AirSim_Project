r"""
export_ppo_onnx.py - PC only: turn the Stage 3 PPO (.zip) into an ONNX file the Raspberry Pi can run
=====================================================================================================
Why: the Pi Zero 2 W (512 MB) runs throw_real.py with onnxruntime + numpy + pymavlink - no PyTorch, no
stable-baselines3 (your plan: "policy runs on the Pi Zero 2, exported via ONNX"). The ONNX file holds the
whole input -> action path: obs_norm.npz scaling + clip, then the PPO actor (13-256-256-128-3, ReLU).
The Pi feeds the 13 raw inputs and gets the 3 actions, exactly as the PPO in the simulator.

1. HOW TO RUN  (Windows PowerShell, folder Q250_Hardware_Throw_Fly)
------------------------------------------------------------------------------------------------------
    py .\export_ppo_onnx.py

    Defaults (no arguments needed): reads  models\round1_final.zip  +  models\obs_norm.npz
                                    writes models\round1_final.onnx
    Needs on the PC: torch, stable-baselines3 (you have them), onnx, onnxruntime:

    py -m pip install onnx onnxruntime

2. EXPECTED RESULT
------------------------------------------------------------------------------------------------------
    | Line printed                                               | Means                                  |
    |------------------------------------------------------------|----------------------------------------|
    | actor: 13 -> 256 -> 256 -> 128 -> 3 (ReLU)                 | the Round 1 network shape              |
    | max difference ONNX vs PPO (200 inputs): 0.000000x (OK)    | < 0.0001 = OK: Pi gives the SAME action|
    | Saved models\round1_final.onnx (~0.4 MB)                   | copy this file to the Pi               |
    "TOO BIG - do not use" = wrong obs_norm.npz or a different network: send me the printout.

3. WHAT IS IN THE .onnx
------------------------------------------------------------------------------------------------------
    | Part                 | What                                                                |
    |----------------------|---------------------------------------------------------------------|
    | input  "obs" [1,13]  | the 13 raw inputs (impact_check.bc_observation)                     |
    | normalise + clip     | (obs - mean) / std, clipped to +/-10 (obs_norm.npz, CLIP_OBS 10)    |
    | actor                | mlp_extractor.policy_net (3 x Linear+ReLU) + action_net             |
    | output "action" [1,3]| roll (rad), pitch (rad), thrust (x hover) - same as the simulator   |
    | metadata "source"    | the .zip name (printed by throw_real.py at start)                   |

4. NEXT
------------------------------------------------------------------------------------------------------
    scp models\round1_final.onnx jlukas@jlukas.local:~/Q250_Hardware_Throw_Fly/models/
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

CLIP_OBS = 10.0            # same as train_imitation.py
LAYERS = ["mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "mlp_extractor.policy_net.4", "action_net"]


def main(args):
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO

    with zipfile.ZipFile(args.model) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    zn = np.load(args.norm)

    class Actor(nn.Module):
        """obs -> normalise + clip -> PPO actor -> action (the whole path, in one graph)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(zn["mean"], dtype=torch.float32))
            self.register_buffer("std", torch.tensor(zn["std"], dtype=torch.float32))
            dims = [sd[f"{n}.weight"].shape for n in LAYERS]
            self.lin = nn.ModuleList([nn.Linear(d[1], d[0]) for d in dims])
            for lin, n in zip(self.lin, LAYERS):
                lin.weight.data.copy_(sd[f"{n}.weight"])
                lin.bias.data.copy_(sd[f"{n}.bias"])

        def forward(self, obs):
            h = torch.clamp((obs - self.mean) / self.std, -CLIP_OBS, CLIP_OBS)
            for lin in self.lin[:3]:
                h = torch.relu(lin(h))
            return self.lin[3](h)

    actor = Actor().eval()
    shape = [actor.lin[0].in_features] + [lin.out_features for lin in actor.lin]
    print(f"actor: {' -> '.join(str(n) for n in shape)} (ReLU)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["obs"], output_names=["action"], opset_version=13)
    try:
        torch.onnx.export(actor, torch.zeros(1, shape[0]), args.out, dynamo=False, **kw)   # newer torch
    except TypeError:
        torch.onnx.export(actor, torch.zeros(1, shape[0]), args.out, **kw)                 # older torch
    m = onnx.load(args.out)
    m.metadata_props.add(key="source", value=Path(args.model).name)
    onnx.save(m, args.out)

    # check: ONNX (what the Pi runs) against the real PPO's own forward pass (mean action, no clipping)
    ppo = PPO.load(args.model, device="cpu")
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.normal(0, 3, size=shape[0]).astype(np.float32)
        xn = np.clip((x - zn["mean"]) / zn["std"], -CLIP_OBS, CLIP_OBS).astype(np.float32)
        with torch.no_grad():
            a_ppo = ppo.policy._predict(torch.as_tensor(xn[None]), deterministic=True)[0].numpy()
        a_onnx = sess.run(None, {"obs": x[None]})[0][0]
        worst = max(worst, float(np.max(np.abs(a_onnx - a_ppo))))
    ok = worst < 1e-4
    print(f"max difference ONNX vs PPO (200 inputs): {worst:.7f}  ({'OK' if ok else 'TOO BIG - do not use'})")
    if not ok:
        Path(args.out).unlink()
        raise SystemExit(1)
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    print(f"Next: scp {args.out} jlukas@jlukas.local:~/Q250_Hardware_Throw_Fly/models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path("models") / "round1_final.zip"))
    ap.add_argument("--norm", default=str(Path("models") / "obs_norm.npz"))
    ap.add_argument("--out", default=str(Path("models") / "round1_final.onnx"))
    main(ap.parse_args())
