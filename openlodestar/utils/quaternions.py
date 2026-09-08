"""Quaternion utilities for OpenLodestar.

Convention: (w, x, y, z) with the scalar part `w` first.
All quaternions representing rotations are unit quaternions (norm = 1).
"""

import numpy as np


def axis_angle_to_quat(axis, angle_rad):
    """Build a unit quaternion from an axis and an angle in radians.

    The input `axis` is normalised internally, so it does not need to
    already be a unit vector.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2
    w = np.cos(half)
    x, y, z = np.sin(half) * axis
    return np.array([w, x, y, z])


def quat_conjugate(q):
    """Return the conjugate of q: (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.array([w, v[0], v[1], v[2]])


def quat_rotate_vector(q, v):
    """Rotate a 3D vector v using the quaternion q via q * v * q_conj."""
    v_q = np.array([0.0, v[0], v[1], v[2]])
    v_rot_q = quat_mul(quat_mul(q, v_q), quat_conjugate(q))
    return v_rot_q[1:]


def quat_derivative(q, omega):
    """Time derivative of an attitude quaternion given body angular velocity.

    Uses the kinematic equation

        q_dot = 0.5 * q * omega_pure

    where ``omega_pure`` is the body angular velocity ``(wx, wy, wz)``
    promoted to a pure quaternion ``(0, wx, wy, wz)`` and ``*`` is the
    Hamilton product.

    Parameters
    ----------
    q : array-like of shape (4,)
        Current attitude quaternion (w, x, y, z), assumed unit norm.
    omega : array-like of shape (3,)
        Body angular velocity in rad/s.

    Returns
    -------
    numpy.ndarray of shape (4,)
        The 4-component time derivative of ``q``.
    """
    omega_pure = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_mul(q, omega_pure)
