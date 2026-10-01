"""Actuator models for OpenLodestar.

This module contains rigid-body actuators used by the ADCS loop. The
first one is a reaction wheel: an internal flywheel spun by an electric
motor. By Newton's third law the reaction torque on the wheel appears
as an opposite torque on the spacecraft body.

Real reaction wheels are bounded by two hard physical limits:

* **maximum motor torque**   ``tau_max``   in N m
* **maximum wheel speed**    ``omega_max`` in rad/s

Together they define the storable angular momentum

    h_max = J_w * omega_max      (N m s)

which the wheel can accumulate. Once the wheel reaches ``omega_max``,
any further commanded torque in the same direction is physically not
deliverable; this is the "saturation" that makes the PID integrator
useless for attitude control (see docs/notes).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReactionWheel:
    """Specification and (mutable) spin rate of one reaction wheel.

    The wheel spin axis ``axis`` is a unit vector expressed in the
    spacecraft body frame. It is normalised in ``__post_init__``.

    Attributes
    ----------
    inertia : float
        Wheel moment of inertia around its spin axis, in kg m^2.
    max_torque : float
        Maximum torque the motor can apply, in N m. Non-negative.
    max_speed : float
        Maximum wheel spin rate (magnitude), in rad/s. Non-negative.
    axis : numpy.ndarray of shape (3,)
        Unit vector of the wheel spin axis, in the body frame.
    """

    inertia: float
    max_torque: float
    max_speed: float
    axis: np.ndarray

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("ReactionWheel.axis must be a non-zero vector.")
        self.axis = axis / norm
        if self.inertia <= 0.0:
            raise ValueError("ReactionWheel.inertia must be strictly positive.")
        if self.max_torque < 0.0:
            raise ValueError("ReactionWheel.max_torque must be non-negative.")
        if self.max_speed < 0.0:
            raise ValueError("ReactionWheel.max_speed must be non-negative.")

    @property
    def max_momentum(self):
        """Maximum wheel angular momentum |J_w * omega_max| in N m s."""
        return self.inertia * self.max_speed


def wheel_actual_torque(wheel, wheel_speed, tau_cmd):
    """Torque the motor can actually deliver, after applying limits.

    Applies two clips to the commanded torque:

    1. Motor torque saturation: ``|tau| <= wheel.max_torque``.
    2. Momentum saturation: if the wheel is already at its max speed in
       one direction, further torque in that same direction is set to
       zero because the wheel physically cannot accelerate any more.

    Parameters
    ----------
    wheel : ReactionWheel
        Wheel specification.
    wheel_speed : float
        Current wheel spin rate ``Omega`` in rad/s, signed.
    tau_cmd : float
        Torque the controller asks the motor to apply, in N m, signed
        along the wheel's spin axis (positive means accelerate in the
        +axis direction).

    Returns
    -------
    float
        Actual torque delivered by the motor, in N m, along the wheel
        spin axis. Can be less than ``tau_cmd`` in magnitude, and zero
        if the wheel is saturated in the commanded direction.
    """
    # 1. Motor torque limit
    tau = float(np.clip(tau_cmd, -wheel.max_torque, wheel.max_torque))

    # 2. Momentum saturation: no more acceleration once at max speed
    if wheel_speed >= wheel.max_speed and tau > 0.0:
        tau = 0.0
    elif wheel_speed <= -wheel.max_speed and tau < 0.0:
        tau = 0.0

    return tau


def wheel_torque_on_spacecraft(wheel, wheel_speed, tau_cmd):
    """Reaction torque exerted by the wheel on the spacecraft body.

    By Newton's third law this is the opposite of the torque the motor
    applies to the wheel, projected onto the wheel spin axis expressed
    in the body frame.

    Parameters
    ----------
    wheel : ReactionWheel
    wheel_speed : float
    tau_cmd : float

    Returns
    -------
    numpy.ndarray of shape (3,)
        Reaction torque on the spacecraft body, in body frame, in N m.
    """
    tau_actual = wheel_actual_torque(wheel, wheel_speed, tau_cmd)
    return -tau_actual * wheel.axis
