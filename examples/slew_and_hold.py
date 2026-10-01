"""Slew-and-hold demo: PD-on-quaternion control with three reaction wheels.

Shows the Phase 2 closed loop in action. A CubeSat starts at the
identity attitude, at rest, and the PD controller is told to slew it to
a tilted target attitude and hold.

The script runs a 20 s simulation and plots:

- pointing error (deg) vs time,
- body-frame angular velocity components,
- reaction-wheel spin rates,
- controller-commanded body torque.

Everything is driven by the dynamics + control modules already in the
package; this file only sets up the scenario and the plots.
"""

# --- Path bootstrap: works without editable install --------------------
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from openlodestar.actuators import ReactionWheel
from openlodestar.control import (
    allocate_to_orthogonal_wheels,
    attitude_error_quaternion,
    pd_torque_on_body,
)
from openlodestar.dynamics import state_derivative_with_wheels
from openlodestar.utils.quaternions import axis_angle_to_quat


# --- Spacecraft configuration ------------------------------------------
inertia_diag = np.array([0.10, 0.12, 0.15])

wheels = [
    ReactionWheel(inertia=1.0e-4, max_torque=1.0e-2,
                  max_speed=800.0, axis=axis)
    for axis in np.eye(3)
]


# --- Initial state and target -----------------------------------------
q0 = np.array([1.0, 0.0, 0.0, 0.0])          # identity attitude
omega0 = np.zeros(3)                          # at rest
wheel_speeds0 = np.zeros(3)                   # wheels at rest
y0 = np.concatenate([q0, omega0, wheel_speeds0])

# Target: 45 deg rotation around (1, 1, 1).
q_target = axis_angle_to_quat(axis=[1, 1, 1], angle_rad=np.deg2rad(45))


# --- PD controller gains ----------------------------------------------
# Chosen for a natural frequency ~0.3 rad/s and damping ratio ~0.7
# against the spacecraft inertia (critical-ish response).
KP = 0.030     # N m  per unit of q_e_vec component
KD = 0.070     # N m s per unit of body angular velocity


# --- Controller as a callable for the integrator ----------------------
def wheel_commands(t, y):
    """PD-on-quaternion + allocation to the three orthogonal wheels."""
    q_current = y[:4]
    omega = y[4:7]
    tau_body = pd_torque_on_body(q_current, omega, q_target, KP, KD)
    return allocate_to_orthogonal_wheels(tau_body, wheels)


# --- Integrate --------------------------------------------------------
t_end = 20.0
sol = solve_ivp(
    state_derivative_with_wheels,
    t_span=(0.0, t_end),
    y0=y0,
    args=(inertia_diag, wheels, wheel_commands),
    method="RK45",
    dense_output=True,
    rtol=1e-9,
    atol=1e-12,
)
if not sol.success:
    raise RuntimeError(f"Integration failed: {sol.message}")

t = np.linspace(0.0, t_end, 1000)
y = sol.sol(t)
q = y[:4]
omega = y[4:7]
wheel_speeds = y[7:]


# --- Post-processing: pointing error, commanded torque ----------------
# Pointing error = total angle between current and target attitude.
pointing_error_deg = np.empty(t.size)
tau_body_history = np.empty((3, t.size))
for k in range(t.size):
    q_e = attitude_error_quaternion(q[:, k], q_target)
    pointing_error_deg[k] = np.rad2deg(2.0 * np.arccos(np.clip(q_e[0], -1.0, 1.0)))
    tau_body_history[:, k] = pd_torque_on_body(
        q[:, k], omega[:, k], q_target, KP, KD,
    )


# --- Summary metrics printed to the console ---------------------------
settling_threshold_deg = 1.0
within = np.where(pointing_error_deg < settling_threshold_deg)[0]
settling_time = t[within[0]] if within.size > 0 else float("nan")

print(f"Final pointing error:  {pointing_error_deg[-1]:.4f} deg")
print(f"Settling time (< {settling_threshold_deg} deg):  "
      f"{settling_time:.2f} s"
      if not np.isnan(settling_time)
      else "Did not settle within tolerance in the simulated window.")
print(f"Peak wheel speed used: "
      f"{np.max(np.abs(wheel_speeds)):.1f} rad/s  "
      f"(limit {wheels[0].max_speed:.0f} rad/s)")


# --- Plots ------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
ax_err, ax_w, ax_wheel, ax_tau = axes

ax_err.plot(t, pointing_error_deg, color="black")
ax_err.axhline(settling_threshold_deg, color="grey", linestyle=":",
               label=f"{settling_threshold_deg}°")
ax_err.set_ylabel("pointing error (deg)")
ax_err.set_title("Slew-and-hold demo - PD on quaternion with 3 reaction wheels")
ax_err.legend(loc="upper right")
ax_err.grid(True, alpha=0.3)

for i, (label, color) in enumerate(zip("xyz", ("red", "green", "blue"))):
    ax_w.plot(t, omega[i], color=color, label=f"w{label}")
ax_w.set_ylabel("body omega (rad/s)")
ax_w.legend(loc="upper right")
ax_w.grid(True, alpha=0.3)

for i, (label, color) in enumerate(zip("xyz", ("red", "green", "blue"))):
    ax_wheel.plot(t, wheel_speeds[i], color=color, label=f"wheel {label}")
ax_wheel.set_ylabel("wheel speed (rad/s)")
ax_wheel.legend(loc="upper right")
ax_wheel.grid(True, alpha=0.3)

for i, (label, color) in enumerate(zip("xyz", ("red", "green", "blue"))):
    ax_tau.plot(t, tau_body_history[i], color=color, label=f"tau_{label}")
ax_tau.set_xlabel("time (s)")
ax_tau.set_ylabel("commanded body torque (N m)")
ax_tau.legend(loc="upper right")
ax_tau.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
