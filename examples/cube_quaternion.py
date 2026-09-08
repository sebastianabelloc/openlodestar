"""Animate a 3D cube rotating around the z-axis using a quaternion.

Visually identical to `cube_matrix.py`; the point is precisely that they
match, showing that a rotation matrix and a unit quaternion encode the
same rotation.

Uses the quaternion utilities from the OpenLodestar package.
"""

# --- Path bootstrap: make the example runnable without needing the
# openlodestar package installed. Adds the repo root to sys.path so
# `from openlodestar...` always resolves.
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from openlodestar.utils.quaternions import (
    axis_angle_to_quat,
    quat_rotate_vector,
)


# --- Cube geometry ------------------------------------------------------
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
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('OpenLodestar - cube rotated with a quaternion')

# Inertial frame axes (fixed)
ax.plot([-1, 1], [0, 0], [0, 0], color='red',   linewidth=2, linestyle='--')
ax.plot([0, 0], [-1, 1], [0, 0], color='green', linewidth=2, linestyle='--')
ax.plot([0, 0], [0, 0], [-1, 1], color='blue',  linewidth=2, linestyle='--')

# Cube edges
lines = []
for (i, j) in edges:
    line, = ax.plot(
        [vertices[i, 0], vertices[j, 0]],
        [vertices[i, 1], vertices[j, 1]],
        [vertices[i, 2], vertices[j, 2]],
        color='steelblue', linewidth=2,
    )
    lines.append(line)

# Body frame axes (rotate with the cube)
body_axes_endpoints = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
])
body_colors = ['red', 'green', 'blue']
body_lines = []
for i in range(3):
    endpoint = body_axes_endpoints[i]
    line, = ax.plot(
        [0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
        color=body_colors[i], linewidth=3, linestyle='-',
    )
    body_lines.append(line)


# --- Per-frame update --------------------------------------------------
def update(frame):
    angle = np.deg2rad(frame)
    q = axis_angle_to_quat(axis=[0, 0, 1], angle_rad=angle)

    rotated = np.array([quat_rotate_vector(q, v) for v in vertices])
    for k, (i, j) in enumerate(edges):
        lines[k].set_data_3d(
            [rotated[i, 0], rotated[j, 0]],
            [rotated[i, 1], rotated[j, 1]],
            [rotated[i, 2], rotated[j, 2]],
        )

    rotated_endpoints = np.array([
        quat_rotate_vector(q, e) for e in body_axes_endpoints
    ])
    for i in range(3):
        endpoint = rotated_endpoints[i]
        body_lines[i].set_data_3d(
            [0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
        )

    return lines + body_lines


anim = FuncAnimation(fig, update, frames=360, interval=30, blit=False)
plt.show()
