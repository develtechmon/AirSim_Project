# Stage 2 Summary: Hover in Wind (ArduPilot SITL + AirSim) - RUN1

**Run folder:** `runs\run1`  
**Generated:** 2026-10-22 16:27  
**Stage 1 network:** `..\Stage_1\runs\run1`  
**Method:** PPO with wind gusts 0-20 m/s (new gust every 1-3 s); warm start from the Stage 1 network vs cold start from random weights, same settings otherwise

## Executive summary

| Check | Status | Key result | Rule |
| --- | --- | --- | --- |
| Warm start loads Stage 1 | **PASS** | max action difference 0.0e+00 | warm PPO acts exactly like Stage 1 at start (< 1e-5) |
| Training, warm start | **PASS** | 501,760 steps, 979 episodes, last 50 success 100% | training finished |
| Training, cold start | **NOT RUN** | - | training finished |
| Warm survives wind | **PASS** | warm success 100%, RMSE 0.198 m | warm survives >= 80% of gust episodes (original target) |
| Better than Stage 1 in wind | **YES** | RMSE warm 0.198 vs Stage 1 0.367 m | Stage 2 holds position better than Stage 1 in wind |
| Better than stock ArduPilot | **YES** | RMSE warm 0.198 vs ArduPilot 0.316 m | Stage 2 holds position better than stock ArduPilot in wind |
| Warm learns faster than cold | **NOT RUN** | - | warm start learns faster than cold start |

## Learning: warm vs cold start

| Measure | Warm start | Cold start |
| --- | --- | --- |
| Episodes | 979 | - |
| Timesteps | 501,713 | - |
| Success, first 50 episodes | 100% | - |
| Mean distance, first 50 episodes | 0.38 m | - |
| Success, last 50 episodes | 100% | - |
| Mean distance, last 50 episodes | 0.14 m | - |
| Steps until rolling success >= 80% | 10,741 | - |

Learning curves: `runs\run1\results\summary\learning_curves_warm_vs_cold.png`

## In wind: all controllers, identical conditions (`stage2_statistics_20261022_161620.json`)

Gusts 0-20 m/s:

| Controller | Success | RMSE (m) | Max (m) | Held within 0.5 m |
| --- | --- | --- | --- | --- |
| PID teacher | 100% | 0.358 | 0.840 | 85% |
| Stage 1 (BC) | 100% | 0.367 | 0.822 | 84% |
| PPO warm | 100% | 0.198 | 0.766 | 96% |
| Stock ArduPilot | 100% | 0.316 | 0.835 | 91% |

Constant wind, RMSE (m) / success:

| Controller | 5 m/s | 10 m/s | 15 m/s | 20 m/s |
| --- | --- | --- | --- | --- |
| PID teacher | 0.167 / 100% | 0.238 / 100% | 0.182 / 100% | 0.338 / 100% |
| Stage 1 (BC) | 0.198 / 100% | 0.257 / 100% | 0.197 / 100% | 0.368 / 100% |
| PPO warm | 0.176 / 100% | 0.232 / 100% | 0.168 / 100% | 0.354 / 100% |
| Stock ArduPilot | 0.159 / 100% | 0.246 / 100% | 0.202 / 100% | 0.353 / 100% |

Stage 2 (warm) vs Stage 1 in gusts: RMSE 0.198 vs 0.367 m (+46.0%).
Stage 2 (warm) vs stock ArduPilot in gusts: RMSE 0.198 vs 0.316 m (+37.4%), held within 0.5 m 96% vs 91%.

## Settings

- PPO: {"lr": 3e-05, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "target_kl": 0.05, "log_std_init": -1.2, "net_arch": "13-256-256-128-3 ReLU (pi and vf)", "critic_warmup_steps": 20000}
- Environment: {"wind": "0-20.0 m/s random direction, vertical x0.3, new gust every 1-3 s", "episode_steps": 500, "hz": 20, "fail_dist_m": 4.0, "reward": "+1 - dist - 0.1*speed - 0.01*|a|^2, -50 on failure", "obs_norm": "..\\Stage_1\\runs\\run1\\models\\obs_norm.npz"}
