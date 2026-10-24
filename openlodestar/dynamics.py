"""Rigid-body rotational dynamics for OpenLodestar.

Currently limited to the case of a diagonal inertia tensor (principal
axes aligned with the body frame). This covers the vast majority of
CubeSat modelling needs and keeps the algebra transparent.
"""

import numpy as np

from openlodestar.actuators import wheel_actual_torque
from openlodestar.quaternions import quaternion_derivative, rotate_vector


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

    q_dot = quaternion_derivative(q, omega)
    omega_dot = angular_acceleration(omega, inertia_diag, torque=tau)

    return np.concatenate([q_dot, omega_dot])


def wheel_momentum_body(wheels, wheel_speeds):
    """Total angular momentum stored in the wheels, expressed in body frame.

    h_wheels = sum_i J_wi * Omega_i * axis_i

    Parameters
    ----------
    wheels : sequence of ReactionWheel
    wheel_speeds : array-like of shape (n_wheels,)
        Current wheel spin rates in rad/s.

    Returns
    -------
    numpy.ndarray of shape (3,)
    """
    h = np.zeros(3)
    for wheel, speed in zip(wheels, wheel_speeds):
        h = h + wheel.inertia * speed * wheel.axis
    return h


def total_angular_momentum_body(inertia_diag, omega, wheels, wheel_speeds):
    """Total spacecraft-plus-wheels angular momentum in body frame.

    h_total_body = I . omega + sum_i J_wi * Omega_i * axis_i
    """
    inertia_diag = np.asarray(inertia_diag, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return inertia_diag * omega + wheel_momentum_body(wheels, wheel_speeds)


def state_derivative_with_wheels(
    t, y, inertia_diag, wheels, wheel_torque_cmds, external_torque_func=None,
):
    """Derivative of the coupled spacecraft-plus-wheels state.

    The state vector packs attitude, body angular velocity and each
    wheel's spin rate

        y = [q_w, q_x, q_y, q_z, wx, wy, wz, Omega_1, ..., Omega_n]

    The dynamics are:

        q_dot     = 0.5 * q * omega_pure
        omega_dot = I_inv . (tau_ext - sum_i tau_actual_i * axis_i
                             - omega x (I . omega + h_wheels))
        Omega_i_dot = tau_actual_i / J_wi

    where ``tau_actual_i`` is the torque actually delivered by wheel i's
    motor once torque and momentum saturation are applied.

    Parameters
    ----------
    t : float
    y : array-like of shape (7 + n_wheels,)
    inertia_diag : array-like of shape (3,)
        Diagonal spacecraft inertia (including wheels treated as static
        masses), in kg m^2.
    wheels : sequence of ReactionWheel
    wheel_torque_cmds : callable(t, y) -> array-like of shape (n_wheels,)
        Motor torque commanded to each wheel (before saturation).
    external_torque_func : callable(t, y) -> array-like of shape (3,) or None
        External torque on the spacecraft body, N m. Default: zero.

    Returns
    -------
    numpy.ndarray of shape (7 + n_wheels,)
    """
    n_wheels = len(wheels)
    q = y[:4]
    omega = y[4:7]
    wheel_speeds = y[7:7 + n_wheels]

    inertia_diag = np.asarray(inertia_diag, dtype=float)

    tau_ext = (
        np.zeros(3) if external_torque_func is None
        else np.asarray(external_torque_func(t, y), dtype=float)
    )

    tau_cmds = np.asarray(wheel_torque_cmds(t, y), dtype=float)

    # Actual torque each motor manages to apply to its own wheel
    tau_actuals = np.array([
        wheel_actual_torque(wheel, speed, tau_cmd)
        for wheel, speed, tau_cmd in zip(wheels, wheel_speeds, tau_cmds)
    ])

    # Reaction torque on the spacecraft body (Newton III):
    # tau_body_from_wheels = - sum_i tau_actual_i * axis_i
    wheel_reaction = np.zeros(3)
    for tau, wheel in zip(tau_actuals, wheels):
        wheel_reaction = wheel_reaction + tau * wheel.axis

    # Wheel angular momentum in body frame
    h_wheels = wheel_momentum_body(wheels, wheel_speeds)

    # Body angular acceleration (Euler with wheels)
    I_omega = inertia_diag * omega
    gyroscopic = np.cross(omega, I_omega + h_wheels)
    omega_dot = (tau_ext - wheel_reaction - gyroscopic) / inertia_diag

    # Wheel spin acceleration
    wheel_inertias = np.array([w.inertia for w in wheels])
    wheel_speeds_dot = tau_actuals / wheel_inertias

    # Quaternion kinematics
    q_dot = quaternion_derivative(q, omega)

    return np.concatenate([q_dot, omega_dot, wheel_speeds_dot])
