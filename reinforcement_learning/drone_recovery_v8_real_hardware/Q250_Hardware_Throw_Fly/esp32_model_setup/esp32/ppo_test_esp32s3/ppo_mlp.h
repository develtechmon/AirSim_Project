/* ppo_mlp.h - the Round 1 PPO actor in plain C (same maths as round1_final.onnx, see export_ppo_esp32.py) */
#pragma once
/* obs[13] = the same 13 inputs as the Pi (impact_check.bc_observation); act[3] = roll rad, pitch rad, thrust x hover */
void ppo_act(const float obs[13], float act[3]);
