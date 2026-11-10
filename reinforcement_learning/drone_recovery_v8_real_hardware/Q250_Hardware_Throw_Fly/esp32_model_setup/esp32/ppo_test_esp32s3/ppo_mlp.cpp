/* ppo_mlp.cpp - Round 1 PPO actor: clip((obs-mean)/std, +-10) -> 13-256-256-128-3 (ReLU) -> action.
   Weights: int16 + one float scale per output row (ppo_weights.h, made by export_ppo_esp32.py), in fast RAM. */
#include <stdint.h>
#include "ppo_mlp.h"
#include "ppo_weights.h"

static const float CLIP_OBS = 10.0f;

static void dense(const int16_t *w, const float *scale, const float *b, const float *x, float *y,
                  int n_in, int n_out, bool relu)
{
    for (int o = 0; o < n_out; o++) {
        const int16_t *row = w + o * n_in;          /* PyTorch Linear: weight[out][in] */
        float s = 0.0f;
        for (int i = 0; i < n_in; i++) s += (float)row[i] * x[i];
        s = s * scale[o] + b[o];
        y[o] = (relu && s < 0.0f) ? 0.0f : s;
    }
}

void ppo_act(const float obs[13], float act[3])
{
    static float x[13], h1[256], h2[256], h3[128];
    for (int i = 0; i < 13; i++) {
        float v = (obs[i] - PPO_MEAN[i]) / PPO_STD[i];
        x[i] = v > CLIP_OBS ? CLIP_OBS : (v < -CLIP_OBS ? -CLIP_OBS : v);
    }
    dense(PPO_Q0, PPO_S0, PPO_B0, x, h1, 13, 256, true);
    dense(PPO_Q1, PPO_S1, PPO_B1, h1, h2, 256, 256, true);
    dense(PPO_Q2, PPO_S2, PPO_B2, h2, h3, 256, 128, true);
    dense(PPO_Q3, PPO_S3, PPO_B3, h3, act, 128, 3, false);
}
