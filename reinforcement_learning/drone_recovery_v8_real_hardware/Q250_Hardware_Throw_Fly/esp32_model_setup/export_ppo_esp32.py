r"""
export_ppo_esp32.py - PC only: turn the PPO ONNX file into C files the ESP32-S3 (Arduino IDE) can run
=====================================================================================================
Why: the ESP32-S3 cannot run onnxruntime. The network is small (13-256-256-128-3, 102,685 numbers), so we
write its numbers into C header files and do the same maths in plain C (ppo_mlp.cpp). Same idea as
export_ppo_onnx.py made the .onnx for the Pi: one script, one model in, files out, checked.

1. HOW TO RUN  (Windows PowerShell, folder Q250_Hardware_Throw_Fly)
------------------------------------------------------------------------------------------------------
    py .\export_ppo_esp32.py

    Defaults (no arguments needed): reads  models\round1_final.onnx   (made by export_ppo_onnx.py)
                                    writes into BOTH Arduino sketch folders (next to this script):
                                      esp32\ppo_test_esp32s3\   (phase 1: the PPO test)
                                      esp32\throw_esp32s3\      (phase 2: the throw catcher)
    Needs on the PC: numpy, onnx, onnxruntime (you have them from export_ppo_onnx.py).

2. EXPECTED RESULT
------------------------------------------------------------------------------------------------------
    | Line printed                                                     | Means                           |
    |------------------------------------------------------------------|---------------------------------|
    | network: 13 -> 256 -> 256 -> 128 -> 3 (ReLU), source round1_...  | the Round 1 PPO                 |
    | int16 vs ONNX (10000 inputs): max 0.00005 (OK)                   | < 0.001 = the ESP32 gives the   |
    |                                                                  | same actions as the Pi          |
    | wrote ...ppo_test_esp32s3\ppo_weights.h (753 KB) + test_vectors.h | copy done, open the sketch      |
    | wrote ...throw_esp32s3\ppo_weights.h (753 KB)                    |                                 |
    "TOO BIG - do not use" = wrong .onnx or a different network: send me the printout.

3. WHAT IS WRITTEN  (in each sketch folder)
------------------------------------------------------------------------------------------------------
    | File             | What                                                                         |
    |------------------|------------------------------------------------------------------------------|
    | ppo_weights.h    | obs_norm mean/std + biases (float32) + weights as int16 with one scale per   |
    |                  | output row (weight = q x scale). 200 KB, kept in the ESP32's fast RAM.       |
    |                  | Why int16: float32 would be 411 KB and must be read from slow flash every    |
    |                  | step (~8-15 ms); int16 fits in RAM (~3-5 ms) and differs by < 0.0001.        |
    | test_vectors.h   | (ppo_test_esp32s3 only) 200 inputs + the ONNX answers for them: the test     |
    |                  | sketch checks itself on the S3 (100 realistic, 50 wide, 50 beyond the clip)  |

4. WHAT THE C CODE COMPUTES  (ppo_mlp.cpp - the same graph as the .onnx)
------------------------------------------------------------------------------------------------------
    x = clip((obs - mean) / std, -10, 10)  ->  h1 = ReLU(W0 x + b0)  ->  h2 = ReLU(W1 h1 + b1)
    ->  h3 = ReLU(W2 h2 + b2)  ->  action = W3 h3 + b3      (roll rad, pitch rad, thrust x hover)

5. NEXT
------------------------------------------------------------------------------------------------------
    Arduino IDE: open esp32\ppo_test_esp32s3\ppo_test_esp32s3.ino -> Upload -> Serial Monitor 115200
    (ESP32-S3 guide, step 2).
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SKETCHES = [HERE / "esp32" / "ppo_test_esp32s3", HERE / "esp32" / "throw_esp32s3"]
LAYERS = ["lin.0", "lin.1", "lin.2", "lin.3"]
CLIP_OBS = 10.0


def c_array(ctype, name, values, fmt):
    v = np.asarray(values).ravel()
    lines = [", ".join(fmt(x) for x in v[i:i + 12]) for i in range(0, len(v), 12)]
    return f"PPO_RAM static const {ctype} {name}[{len(v)}] = {{\n" + ",\n".join(lines) + "\n};\n"


def f32(x):
    return f"{float(x):.9e}f"


def forward(W, X, int16=None):
    h = np.clip((X - W["mean"]) / W["std"], -CLIP_OBS, CLIP_OBS)
    for i, n in enumerate(LAYERS):
        w = W[f"{n}.weight"] if int16 is None else int16[n]
        h = h @ w.T + W[f"{n}.bias"]
        if i < 3:
            h = np.maximum(h, 0.0)
    return h


def main(args):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    m = onnx.load(args.onnx)
    W = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in m.graph.initializer}
    source = {p.key: p.value for p in m.metadata_props}.get("source", Path(args.onnx).name)
    shape = [W["lin.0.weight"].shape[1]] + [W[f"{n}.weight"].shape[0] for n in LAYERS]
    print(f"network: {' -> '.join(map(str, shape))} (ReLU), source {source}")

    # int16 weights, one scale per output row
    Q, S, deq = {}, {}, {}
    for n in LAYERS:
        w = W[f"{n}.weight"]
        s = (np.abs(w).max(axis=1) / 32767.0).astype(np.float32)
        s[s == 0] = 1.0
        q = np.clip(np.round(w / s[:, None]), -32767, 32767).astype(np.int16)
        Q[n], S[n], deq[n] = q, s, q.astype(np.float32) * s[:, None]

    # check against onnxruntime (what the Pi runs)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mean, std = W["mean"], W["std"]
    X = np.concatenate([mean + std * rng.normal(0, 1, (5000, 13)), rng.normal(0, 3, (4000, 13)),
                        mean + std * rng.uniform(-30, 30, (1000, 13))]).astype(np.float32)
    ref = np.concatenate([sess.run(None, {"obs": X[i:i + 1]})[0] for i in range(len(X))])
    worst = float(np.abs(forward(W, X, deq) - ref).max())
    ok = worst < 1e-3
    print(f"int16 vs ONNX ({len(X)} inputs): max {worst:.5f} ({'OK' if ok else 'TOO BIG - do not use'})")
    if not ok:
        raise SystemExit(1)

    idx = np.r_[0:100, 5000:5050, 9000:9050]
    head = ("/* ppo_weights.h - made by export_ppo_esp32.py from %s (source %s). Do not edit by hand. */\n"
            "#pragma once\n#include <stdint.h>\n#ifdef ARDUINO\n#include \"esp_attr.h\"\n#define PPO_RAM DRAM_ATTR\n"
            "#else\n#define PPO_RAM\n#endif\n#define PPO_SOURCE \"%s\"\n" % (Path(args.onnx).name, source, source))
    body = c_array("float", "PPO_MEAN", mean, f32) + c_array("float", "PPO_STD", std, f32)
    for i, n in enumerate(LAYERS):
        body += c_array("int16_t", f"PPO_Q{i}", Q[n], lambda x: str(int(x)))
        body += c_array("float", f"PPO_S{i}", S[n], f32)
        body += c_array("float", f"PPO_B{i}", W[f"{n}.bias"], f32)
    tv = ("/* test_vectors.h - made by export_ppo_esp32.py: inputs + the ONNX answers (what the Pi computes) */\n"
          "#pragma once\n#ifndef PPO_RAM\n#define PPO_RAM\n#endif\n#define PPO_N_TEST %d\n" % len(idx)
          + c_array("float", "PPO_TEST_OBS", X[idx], f32).replace("PPO_RAM ", "")
          + c_array("float", "PPO_TEST_ACT", ref[idx], f32).replace("PPO_RAM ", ""))
    for d in [Path(p) for p in (args.out or SKETCHES)]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "ppo_weights.h").write_text(head + body)
        extra = ""
        if d.name == "ppo_test_esp32s3":                    # the test inputs only for the phase-1 test sketch
            (d / "test_vectors.h").write_text(tv)
            extra = " + test_vectors.h"
        print(f"wrote {d / 'ppo_weights.h'} ({(d / 'ppo_weights.h').stat().st_size / 1e3:.0f} KB){extra}")
    print("Next: Arduino IDE -> open esp32\\ppo_test_esp32s3\\ppo_test_esp32s3.ino -> Upload -> Serial Monitor 115200")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(HERE / "models" / "round1_final.onnx"))
    ap.add_argument("--out", nargs="*", help="sketch folders (default: both esp32 sketches)")
    main(ap.parse_args())
