r"""
CURRICULUM LEVEL COMPARISON - "NO CATASTROPHIC FORGETTING"  (step 3d, ArduPilot SITL + AirSim)
=============================================================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running)
------------------------------------------------------------------------
    Needs the FROZEN final model set in models\stage3_final\ (see train_stage_3_gated_curriculum.py, FINAL
    METHOD): gated_curriculum_policy.zip, level_0_EASY_mastered.zip, level_1_MEDIUM_mastered.zip.
    Yaw brake ON (0.5 s) by default. Another folder (e.g. a new training): --models-dir models\stage3_checkpoints\curriculum_levels

    py .\compare_curriculum_model.py

    More hits per level (5 per type and level, ~75 min):

    py .\compare_curriculum_model.py --repeats 5


2. EXPECTED RESULT
------------------------------------------------------------------------
    Table: every model on EVERY level (same hits), e.g.
        Model           Level    Recovered  Avg lowest ...
        bc-policy       EASY       100 %      ~22 m
        ppo-level0      EASY       100 %      >= BC
        ppo-level1      ...
        ppo-final       EASY       100 %      >= ppo-level0    <- no forgetting of EASY
        ppo-final       HARD       100 %      highest          <- learned HARD
    How to read it:
    Result                                           Meaning
    ppo-final >= earlier models on EASY and MEDIUM   no catastrophic forgetting (thesis Table 5.3)
    ppo-final worse on EASY than ppo-level0          some forgetting - report it honestly


3. OUTPUT FILES
------------------------------------------------------------------------
    runs\compare_curriculum\compare_curriculum_<date>.csv / _steps.csv / .json   every flight
    analysis_results\curriculum_comparison_<date>.json
    analysis_results\table_5_3_curriculum_comparison.tex
    analysis_results\curriculum_comparison_<date>_lowest.png


4. WHAT THE SCRIPT DOES
------------------------------------------------------------------------
    Step 1  load bc-policy, ppo-level0 (EASY mastered), ppo-level1 (MEDIUM mastered), ppo-final
            (models that do not exist yet are skipped and named)
    Step 2  every hit is flown once by each model (same seed = same hit), all 3 levels
    Step 3  table per model x level, saved for the thesis
"""

import argparse
import datetime
from pathlib import Path

import impact_check as ic
from test_stage_3_gated_curriculum import LEVELS, print_tables, run, save_outputs, summarise

FINAL_DIR = Path("models/stage3_final")    # frozen final model set (see train_stage_3_gated_curriculum.py, FREEZE)
MODELS = {"ppo-level0": FINAL_DIR / "level_0_EASY_mastered.zip",
          "ppo-level1": FINAL_DIR / "level_1_MEDIUM_mastered.zip",
          "ppo-final": FINAL_DIR / "gated_curriculum_policy.zip"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(ic.DISTURBANCES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--yaw-brake", type=float, default=ic.YAW_BRAKE_T,
                    help="yaw brake time constant (s), 0 = off. Default ic.YAW_BRAKE_T: ON for evaluation (final method)")
    ap.add_argument("--models-dir", default=str(FINAL_DIR), help="folder with the 3 models (default: the frozen final set)")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    a = ap.parse_args()
    ic.YAW_BRAKE_T = a.yaw_brake
    d = Path(a.models_dir)
    MODELS = {k: d / v.name for k, v in MODELS.items()}
    modes = ["bc-policy"]
    for name, path in MODELS.items():
        if path.exists():
            ic.register_policy(name, path)
            modes.append(name)
        else:
            print(f"   (skipped {name}: {path} not found)")
    if len(modes) == 1:
        raise SystemExit("No Stage 3 models found - train first (train_stage_3_gated_curriculum.py)")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    impacts = run(modes, a.types, LEVELS, a.repeats, a.seed, a.conn, "compare_curriculum")
    res = summarise(impacts, modes)
    res["settings"] = {"models": {m: str(MODELS.get(m, ic.POLICIES.get(m))) for m in modes},
                       "types": a.types, "repeats": a.repeats, "seed": a.seed, "yaw_brake_t": a.yaw_brake}
    print_tables(res, modes, modes[-1])
    save_outputs(res, modes, stamp, "curriculum_comparison", "table_5_3_curriculum_comparison.tex",
                 "Curriculum models on all levels - no catastrophic forgetting")
