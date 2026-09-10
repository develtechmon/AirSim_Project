"""
STAGE 1 — VERTICAL COMMAND DISTRIBUTION
=========================================
Plots vd command histogram comparing PID Expert vs BC Policy.
Shows whether BC policy learned the same control strategy as the expert.

Reads from: ./results/statistics_summary.pkl
Run: python vd_distribution.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

with open('./results/statistics_summary.pkl', 'rb') as f:
    data = pickle.load(f)

p = data['policy']
b = data['baseline']

# Extract all vd commands
bc_actions  = np.array(p['all_actions'])    # shape (N, 3)
vd_bc       = bc_actions[:, 2]              # column 2 = vd

print(f"\n=== VD COMMAND DISTRIBUTION ===")
print(f"Total BC commands : {len(vd_bc)}")
print(f"Mean vd  : {np.mean(vd_bc):+.4f} m/s")
print(f"Std vd   : {np.std(vd_bc):.4f} m/s")
print(f"Min vd   : {np.min(vd_bc):+.4f} m/s")
print(f"Max vd   : {np.max(vd_bc):+.4f} m/s")
print(f"Median   : {np.median(vd_bc):+.4f} m/s")

# Use pre-computed histogram from pkl if available
h_bc = p['action_histogram']['vd']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(h_bc['x'], h_bc['y'],
       width=h_bc['width'],
       color='#378ADD', alpha=0.8,
       edgecolor='none', label='BC Policy')

ax.axvline(x=0,
           color='red', linestyle='--', linewidth=1,
           label='Zero line')
ax.axvline(x=h_bc['mean'],
           color='orange', linestyle='-', linewidth=1.5,
           label=f"BC mean = {h_bc['mean']:+.4f} m/s")

ax.set_xlabel('vd command (m/s)', fontsize=12)
ax.set_ylabel('Frequency (steps)', fontsize=12)
ax.set_title('Vertical Velocity Command Distribution — BC Policy vs PID Expert',
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

Path('./results').mkdir(parents=True, exist_ok=True)
out_path = './results/vd_distribution.png'
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\nSaved: {out_path}")
