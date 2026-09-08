"""Animate a torque-free tumbling rigid body in 3D.

The physics is identical to ``tumble_simulation.py``: an asymmetric
CubeSat-scale rigid body with an initial angular velocity that is not
aligned with any principal axis, integrated under zero external torque.
The difference is the rendering: instead of plotting omega and q against
time, the attitude ``q(t)`` is used to rotate a 3D cube frame by frame,
so the tumbling is directly visible.

Inertial axes (dashed, transparent) stay fixed in the inertial frame.
Body axes (solid, thicker) rotate with the cube. At t=0 they coincide.

The Phase 1 deliverable of OpenLodestar's roadmap.
"""

# --- Path bootstrap: works without editable install --------------------
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from openlodestar.dynamics import state_derivative
from openlodestar.utils.quaternions import quat_rotate_vector


# --- Spacecraft configuration ------------------------------------------
# Asymmetric CubeSat-scale inertia (kg m^2).
inertia_diag = np.array([0.10, 0.12, 0.15])

# Initial attitude: identity (body frame aligned with inertial frame).
q0 = np.array([1.0, 0.0, 0.0, 0.0])

# Initial angular velocity (rad/s). Not aligned with any principal axis
# so that nutation is visible.
omega0 = np.array([0.4, 0.9, 0.2])

y0 = np.concatenate([q0, omega0])


# --- Simulation (run once, cache q(t)) ---------------------------------
t_end = 20.0
n_frames = 500

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

t_frames = np.linspace(0.0, t_end, n_frames)
y_frames = sol.sol(t_frames)
q_frames = y_frames[:4]   # shape (4, n_frames)


# --- Cube geometry -----------------------------------------------------
vertices = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
])

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# --- Figure setup ------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("OpenLodestar - torque-free tumble (Phase 1)")

# Inertial axes (fixed reference)
ax.plot([-1, 1], [0, 0], [0, 0], color="red",   linewidth=2, linestyle="--", alpha=0.5)
ax.plot([0, 0], [-1, 1], [0, 0], color="green", linewidth=2, linestyle="--", alpha=0.5)
ax.plot([0, 0], [0, 0], [-1, 1], color="blue",  linewidth=2, linestyle="--", alpha=0.5)

# Cube edges (updated each frame)
lines = []
for (i, j) in edges:
    line, = ax.plot(
        [vertices[i, 0], vertices[j, 0]],
        [vertices[i, 1], vertices[j, 1]],
        [vertices[i, 2], vertices[j, 2]],
        color="steelblue", linewidth=2,
    )
    lines.append(line)

# Body-frame axes (rotate with the cube)
body_axes_endpoints = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
])
body_colors = ["red", "green", "blue"]
body_lines = []
for i in range(3):
    endpoint = body_axes_endpoints[i]
    line, = ax.plot(
        [0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
        color=body_colors[i], linewidth=3, linestyle="-",
    )
    body_lines.append(line)

# Time counter overlay (top-left of the plot)
time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)


# --- Per-frame update --------------------------------------------------
def update(frame):
    q = q_frames[:, frame]

    # Rotate cube vertices with the current attitude
    rotated = np.array([quat_rotate_vector(q, v) for v in vertices])
    for k, (i, j) in enumerate(edges):
        lines[k].set_data_3d(
            [rotated[i, 0], rotated[j, 0]],
            [rotated[i, 1], rotated[j, 1]],
            [rotated[i, 2], rotated[j, 2]],
        )

    # Rotate body axes with the same attitude
    rotated_endpoints = np.array([
        quat_rotate_vector(q, e) for e in body_axes_endpoints
    ])
    for i in range(3):
        endpoint = rotated_endpoints[i]
        body_lines[i].set_data_3d(
            [0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
        )

    time_text.set_text(f"t = {t_frames[frame]:6.2f} s")

    return lines + body_lines + [time_text]


anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
plt.show()
