"""
pid_expert.py - Stage 1 expert (teacher) for the ArduPilot pipeline
===================================================================
OBSERVABLE EXPERT: the command depends ONLY on what the network sees.

    velocity command = gain x (target - position), capped

    horizontal: gain 0.8 /s, cap 1.0 m/s
    vertical:   gain 1.0 /s, cap 0.5 m/s

Why this teacher (and not the old cascade PID):
    The old PID's command also depended on hidden memory (integral term,
    derivative of a noisy error, a 5 m/s kick at every new target).
    The network cannot see that memory, so it could not copy it:
    first run gave 42% transfer efficiency (statistics_summary_20261017_140904).
    ArduPilot's own velocity controller (with its own integrator) removes the
    last small error, so the teacher does not need memory of its own.

Same interface as before (PIDExpertHover, get_action, reset), so no other
script changes.
"""

import numpy as np

KP_XY = 0.8      # 1/s
KP_Z = 1.0       # 1/s
VMAX_XY = 1.0    # m/s
VMAX_Z = 0.5     # m/s


class PIDExpertHover:
    """Input: state dict with 'position' = position - target (NED, m).
    Output: velocity command [vx, vy, vz] (NED, m/s)."""

    def __init__(self, dt=0.05):
        self.dt = dt                     # kept for compatibility, not used

    def get_action(self, state):
        rel = np.asarray(state["position"], dtype=np.float64)
        v = np.array([-KP_XY * rel[0], -KP_XY * rel[1], -KP_Z * rel[2]])
        v[0:2] = np.clip(v[0:2], -VMAX_XY, VMAX_XY)
        v[2] = np.clip(v[2], -VMAX_Z, VMAX_Z)
        return v.astype(np.float32)

    def reset(self):
        pass                             # no memory to reset


EXPERT_DESCRIPTION = (f"observable P expert: v = gain x (target - position); "
                      f"xy gain {KP_XY}/s cap {VMAX_XY} m/s, z gain {KP_Z}/s cap {VMAX_Z} m/s")
