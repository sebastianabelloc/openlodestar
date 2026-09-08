"""Rigid-body rotational dynamics for OpenLodestar.

Currently limited to the case of a diagonal inertia tensor (principal
axes aligned with the body frame). This covers the vast majority of
CubeSat modelling needs and keeps the algebra transparent.
"""

import numpy as np

from openlodestar.utils.quaternions import quat_derivative


def angular_acceleration(omega, inertia_diag, torque=None):
    """Body-frame angular acceleration from Euler's rotational equation.

    Uses

        I . omega_dot + omega x (I . omega) = tau

    solved for ``omega_dot``

        omega_dot = I_inv . (tau - omega x (I . omega))

    with a diagonal inertia tensor ``I = diag(Ixx, Iyy, Izz)``. The
    ``omega x (I . omega)`` term is the "gyroscopic" or "coriolis" term
    that arises from expressing Euler's equation in the (rotating) body
    frame.

    Parameters
    ----------
    omega : array-like of shape (3,)
        Body angular velocity in rad/s, expressed in the body frame.
    inertia_diag : array-like of shape (3,)
        Diagonal entries of the inertia tensor: ``(Ixx, Iyy, Izz)``.
        All entries must be strictly positive.
    torque : array-like of shape (3,) or None, optional
        External torque expressed in the body frame, in N m.
        If ``None`` (default), the body is considered torque-free.

    Returns
    -------
    numpy.ndarray of shape (3,)
        Body-frame angular acceleration ``omega_dot`` in rad/s^2.
    """
    omega = np.asarray(omega, dtype=float)
    inertia_diag = np.asarray(inertia_diag, dtype=float)
    tau = np.zeros(3) if torque is None else np.asarray(torque, dtype=float)

    # For a diagonal inertia tensor, I . omega is elementwise multiplication.
    I_omega = inertia_diag * omega
    gyroscopic = np.cross(omega, I_omega)

    return (tau - gyroscopic) / inertia_diag


def state_derivative(t, y, inertia_diag, torque_func=None):
    """Derivative of the combined attitude/rate state for a rigid body.

    The state vector ``y`` packs attitude and body angular velocity as

        y = [q_w, q_x, q_y, q_z, wx, wy, wz]

    Splits the state, evaluates the quaternion kinematic equation
    (``q_dot = 0.5 * q * omega_pure``) and Euler's rotational equation
    (``omega_dot = I_inv . (tau - omega x (I . omega))``), and returns
    the packed derivative in the same layout.

    Parameters
    ----------
    t : float
        Current time in seconds. Passed to ``torque_func`` if given.
    y : array-like of shape (7,)
        Packed state ``[q, omega]``. ``q`` is assumed to be unit norm.
    inertia_diag : array-like of shape (3,)
        Diagonal inertia tensor entries (Ixx, Iyy, Izz) in kg m^2.
    torque_func : callable(t, y) -> (3,) array or None, optional
        External torque expressed in the body frame. If ``None`` (default),
        the body is torque-free.

    Returns
    -------
    numpy.ndarray of shape (7,)
        Packed derivative ``[q_dot, omega_dot]``.
    """
    q = y[:4]
    omega = y[4:]

    tau = None if torque_func is None else torque_func(t, y)

    q_dot = quat_derivative(q, omega)
    omega_dot = angular_acceleration(omega, inertia_diag, torque=tau)

    return np.concatenate([q_dot, omega_dot])
