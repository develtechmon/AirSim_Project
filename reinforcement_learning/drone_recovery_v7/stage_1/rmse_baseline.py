"""
STAGE 1 — RMSE BASELINE COMPARISON
====================================
Prints per-episode RMSE table comparing PID Expert vs BC Policy.
Shows transfer efficiency of behavioural cloning.

Reads from: ./results/statistics_summary.pkl
Run: python rmse_baseline.py
"""

import pickle
import numpy as np

with open('./results/statistics_summary.pkl', 'rb') as f:
    data = pickle.load(f)

b = data['baseline']   # PID Expert
p = data['policy']     # BC Policy
c = data['comparison']
target = data['target_alt']

print("\n" + "="*82)
print("STAGE 1 — RMSE BASELINE COMPARISON")
print(f"PID Expert (AirSim) vs BC Policy | Target: {target}m")
print("="*82)

print("\nTABLE 1 — AGGREGATE METRICS")
print("-"*72)
print(f"{'Metric':<38} {'PID Expert':>12} {'BC Policy':>12} {'Gap':>10}")
print("-"*72)

metrics = [
    ("Mean altitude (m)",          'alt_mean'),
    ("Mean altitude error (m)",    'alt_error_mean'),
    ("RMSE from target (m)",       'alt_rmse_mean'),
    ("Altitude std (m)",           'alt_std_mean'),
    ("Mean horiz. drift (m)",      'dist_mean'),
    ("Max horiz. drift (m)",       'dist_max_mean'),
]
for label, key in metrics:
    bv = b[key]; pv = p[key]
    print(f"  {label:<36} {bv:>12.4f} {pv:>12.4f} {pv-bv:>+10.4f}")

print("-"*72)
print(f"  {'RMSE ratio (policy / baseline)':<36} {'':>12} {c['rmse_ratio']:>12.4f}")
print(f"  {'Transfer efficiency (%)':<36} {'':>12} {c['transfer_efficiency']:>12.2f}")

print("\nTABLE 2 — PER-EPISODE RMSE")
print("-"*82)
print(f"{'Ep':<4} {'BL alt':>8} {'BL std':>7} {'BL RMSE':>8} "
      f"{'BC alt':>8} {'BC std':>7} {'BC RMSE':>8}")
print("-"*82)

for i in range(len(b['ep_means'])):
    print(f"{i+1:<4} {b['ep_means'][i]:>8.3f} {b['ep_stds'][i]:>7.4f} "
          f"{b['ep_rmse'][i]:>8.4f} "
          f"{p['ep_means'][i]:>8.3f} {p['ep_stds'][i]:>7.4f} "
          f"{p['ep_rmse'][i]:>8.4f}")

print("-"*82)
print(f"{'Mean':<4} {np.mean(b['ep_means']):>8.3f} {np.mean(b['ep_stds']):>7.4f} "
      f"{np.mean(b['ep_rmse']):>8.4f} "
      f"{np.mean(p['ep_means']):>8.3f} {np.mean(p['ep_stds']):>7.4f} "
      f"{np.mean(p['ep_rmse']):>8.4f}")

print(f"\nRMSE ratio  : {c['rmse_ratio']:.4f}×")
print(f"Transfer eff: {c['transfer_efficiency']:.2f}%")

print(f"\nInterpretation:")
print(f"  PID Expert hovers at {np.mean(b['ep_means']):.3f}m (target: {target}m)")
print(f"  BC Policy  hovers at {np.mean(p['ep_means']):.3f}m (target: {target}m)")

offset = target - np.mean(b['ep_means'])
if abs(offset) > 0.01:
    print(f"  Barometric/physics offset: {offset:.3f}m below target")
    print(f"  This offset contributes ~{abs(offset):.3f}m to baseline RMSE")

if c['transfer_efficiency'] >= 90:
    print(f"\n  ✅ Transfer efficiency {c['transfer_efficiency']:.1f}% — "
          f"BC policy successfully learned from PID expert")
elif c['transfer_efficiency'] >= 75:
    print(f"\n  ⚠️  Transfer efficiency {c['transfer_efficiency']:.1f}% — "
          f"acceptable but consider more demonstrations")
else:
    print(f"\n  ❌ Transfer efficiency {c['transfer_efficiency']:.1f}% — "
          f"consider retraining with more episodes")

print("="*82)
