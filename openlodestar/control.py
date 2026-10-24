"""Attitude control laws for OpenLodestar.

Currently provides the classic **PD-on-quaternion** law used across the
industry for slew-and-hold pointing tasks.

Convention: attitude quaternions ``(w, x, y, z)`` map body vectors to
inertial vectors, and angular velocity ``omega`` is expressed in the
body frame.
"""

import numpy as np

from openlodestar.quaternions import quaternion_conjugate, quaternion_multiplication


def attitude_error_quaternion(q_current, q_target):
    """Attitude error quaternion, taken as the shortest rotation.

    The error quaternion is defined so that

        q_target = q_current * q_e

    which gives

        q_e = q_current_conjugate * q_target

    and represents the rotation, expressed **in the body frame**, that
    would take the current attitude to the target.

    Because ``q`` and ``-q`` represent the same physical orientation, we
    enforce ``q_e[0] >= 0`` so the controller always chases the shorter
    of the two equivalent rotations. Without this the controller could
    command a full extra revolution to correct a tiny angular error.

    Parameters
    ----------
    q_current, q_target : array-like of shape (4,)
        Unit quaternions ``(w, x, y, z)``.

    Returns
    -------
    numpy.ndarray of shape (4,)
        Error quaternion with the scalar part non-negative.
    """
    q_current = np.asarray(q_current, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    q_e = quaternion_multiplication(quaternion_conjugate(q_current), q_target)
    if q_e[0] < 0.0:
        q_e = -q_e
    return q_e


def pd_torque_on_body(q_current, omega, q_target, kp, kd):
    """Torque the PD-on-quaternion law wants to apply to the body.

        tau_body = Kp * q_e_vec  -  Kd * omega

    where ``q_e_vec`` is the vector part of the short-rotation attitude
    error quaternion.

    Both ``kp`` and ``kd`` accept a scalar (isotropic) or a length-3
    array (per-axis diagonal gains). The returned torque is the value
    the *body* should see; the wheel motor commands are its negatives
    projected on the wheel axes (see ``allocate_to_orthogonal_wheels``).

    Parameters
    ----------
    q_current : array-like of shape (4,)
    omega : array-like of shape (3,)
        Body angular velocity in rad/s.
    q_target : array-like of shape (4,)
    kp : float or array-like of shape (3,)
        Proportional gain(s).
    kd : float or array-like of shape (3,)
        Derivative gain(s).

    Returns
    -------
    numpy.ndarray of shape (3,)
        Desired torque on the spacecraft body, in body frame, in N m.
    """
    q_e = attitude_error_quaternion(q_current, q_target)
    q_e_vec = q_e[1:]

    omega = np.asarray(omega, dtype=float)
    kp = np.asarray(kp, dtype=float)
    kd = np.asarray(kd, dtype=float)

    return kp * q_e_vec - kd * omega


def allocate_to_orthogonal_wheels(body_torque, wheels):
    """Wheel motor commands for a set of orthogonal reaction wheels.

    For each wheel, the motor must apply
    ``-body_torque . axis_i`` so that the Newton III reaction on the
    body along axis_i matches the desired ``body_torque . axis_i``.

    This helper assumes the wheels' spin axes span an orthonormal set
    (typically the three body axes). For non-orthogonal configurations
    a pseudo-inverse allocation matrix is needed instead - to be added
    when we introduce a redundant 4-in-pyramid wheel set later.

    Parameters
    ----------
    body_torque : array-like of shape (3,)
        Torque the controller wants to apply to the body, in body frame.
    wheels : sequence of ReactionWheel
        Wheel specifications. Their axes should be mutually orthogonal.

    Returns
    -------
    numpy.ndarray of shape (n_wheels,)
        Torque to command each wheel's motor, along its own spin axis.
    """
    body_torque = np.asarray(body_torque, dtype=float)
    return np.array([-np.dot(body_torque, wheel.axis) for wheel in wheels])
