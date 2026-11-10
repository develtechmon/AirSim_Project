Made by:
make_stage2_summary.py

The main Stage 2 result. PID teacher, Stage 1 network and Stage 2 PPO all flown
in identical wind (10 gust episodes + constant wind 5/10/15/20)

This stage2_summary is built from:
training_info.json                -- training settings + weight-transfer check (PASS/FAIL)
episodes.csv                      -- learning curves → learning_curves_warm_vs_cold.png
stage2_statistics_<newest>.json   -- the wind comparison table: PID vs Stage 1 vs PPO vs Ardupilot

Needed for result

