"""Simulate a torque-free tumbling rigid body and plot its telemetry.

This is the first dynamic simulation of OpenLodestar. It integrates the
coupled attitude/angular-velocity system

    q_dot     = 0.5 * q * omega_pure
    omega_dot = I_inv . (tau - omega x (I . omega))

with ``tau = 0`` (free body), and plots the angular velocity components
and the quaternion components as functions of time.

Because the inertia tensor is asymmetric and the initial angular velocity
is not aligned with any principal axis, the body exhibits **nutation**:
omega evolves non-trivially even though no torque is applied. Angular
momentum and rotational kinetic energy must remain conserved.
"""

# --- Path bootstrap: make the example runnable even if the openlodestar
# package is not installed (e.g. `pip install -e .` had trouble). Adds the
# repo root to sys.path so `from openlodestar...` always resolves.
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from openlodestar.dynamics import state_derivative


# --- Spacecraft configuration ------------------------------------------
# Asymmetric CubeSat-scale inertia (kg m^2).
inertia_diag = np.array([0.10, 0.12, 0.15])

# Initial attitude: identity (body frame aligned with inertial frame).
q0 = np.array([1.0, 0.0, 0.0, 0.0])

# Initial angular velocity (rad/s). Deliberately not aligned with a
# principal axis so that nutation is visible.
omega0 = np.array([0.4, 0.9, 0.2])

y0 = np.concatenate([q0, omega0])


# --- Integration --------------------------------------------------------
t_end = 20.0           # seconds of simulated time
n_samples = 1000       # points at which to evaluate the solution

sol = solve_ivp(
    state_derivative,
    t_span=(0.0, t_end),
    y0=y0,
    args=(inertia_diag,),
    method="RK45",
    dense_output=True,
    rtol=1e-10,
    atol=1e-12,
)

if not sol.success:
    raise RuntimeError(f"Integration failed: {sol.message}")

t = np.linspace(0.0, t_end, n_samples)
y = sol.sol(t)          # shape (7, n_samples)
q = y[:4]
omega = y[4:]


# --- Sanity checks -----------------------------------------------------
# Quaternion norm should stay close to 1. Numerical integrators drift
# slowly off the unit sphere; we print the worst-case deviation to keep
# an eye on it.
q_norms = np.linalg.norm(q, axis=0)
print(f"quaternion norm min/max over the run: "
      f"{q_norms.min():.10f} / {q_norms.max():.10f}")

# Angular momentum magnitude (in body frame) and kinetic energy should
# also be conserved because tau = 0.
h_body = inertia_diag[:, None] * omega
h_magnitude = np.linalg.norm(h_body, axis=0)
kinetic = 0.5 * np.sum(inertia_diag[:, None] * omega ** 2, axis=0)
print(f"|h| range: {h_magnitude.min():.10f} -> {h_magnitude.max():.10f}")
print(f"T   range: {kinetic.min():.10f} -> {kinetic.max():.10f}")


# --- Plots -------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax_w, ax_q = axes

ax_w.plot(t, omega[0], label="wx", color="red")
ax_w.plot(t, omega[1], label="wy", color="green")
ax_w.plot(t, omega[2], label="wz", color="blue")
ax_w.set_ylabel("angular velocity (rad/s)")
ax_w.set_title("Torque-free tumble - angular velocity components (body frame)")
ax_w.legend(loc="upper right")
ax_w.grid(True, alpha=0.3)

ax_q.plot(t, q[0], label="qw")
ax_q.plot(t, q[1], label="qx")
ax_q.plot(t, q[2], label="qy")
ax_q.plot(t, q[3], label="qz")
ax_q.set_xlabel("time (s)")
ax_q.set_ylabel("quaternion components")
ax_q.set_title("Attitude quaternion")
ax_q.legend(loc="upper right", ncols=4)
ax_q.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
