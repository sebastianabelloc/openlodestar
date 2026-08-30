"""Animate a 3D cube rotating around the z-axis using a rotation matrix.

Draws:
 - The cube (blue edges).
 - The inertial frame axes (dashed, fixed).
 - The body frame axes (solid, thicker, rotating with the cube).

This is one of the two Phase 0 deliverables of OpenLodestar (matrix version).
The companion script `cube_quaternion.py` produces an identical animation
using a quaternion, to make the equivalence between representations visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


# --- Rotation matrix around z ------------------------------------------
def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# --- Figure setup ------------------------------------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('OpenLodestar - cube rotated with a rotation matrix')

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

# Body frame axes (rotate with the cube; identical to inertial at t=0)
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
    R = Rz(angle)

    rotated = vertices @ R.T
    for k, (i, j) in enumerate(edges):
        lines[k].set_data_3d(
            [rotated[i, 0], rotated[j, 0]],
            [rotated[i, 1], rotated[j, 1]],
            [rotated[i, 2], rotated[j, 2]],
        )

    rotated_endpoints = body_axes_endpoints @ R.T
    for i in range(3):
        endpoint = rotated_endpoints[i]
        body_lines[i].set_data_3d(
            [0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
        )

    return lines + body_lines


anim = FuncAnimation(fig, update, frames=360, interval=30, blit=False)
plt.show()
