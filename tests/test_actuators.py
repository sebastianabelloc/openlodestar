"""Tests for openlodestar.actuators.

Run from the repo root with:
    python -m pytest
"""

import numpy as np
import pytest

from openlodestar.actuators import (
    ReactionWheel,
    wheel_actual_torque,
    wheel_torque_on_spacecraft,
)


# --- A typical CubeSat wheel: light, low torque, moderate max speed ----
def make_wheel(axis=(0.0, 0.0, 1.0), initial_speed=0.0):
    return ReactionWheel(
        inertia=1.0e-4,      # kg m^2
        max_torque=1.0e-3,   # N m
        max_speed=838.0,     # rad/s  (~8000 rpm)
        axis=np.array(axis),
    )


# --- Construction and validation --------------------------------------

def test_axis_is_normalised_on_construction():
    """A non-unit axis vector must be normalised internally."""
    wheel = ReactionWheel(
        inertia=1e-4, max_torque=1e-3, max_speed=800.0,
        axis=np.array([0.0, 0.0, 5.0]),
    )
    assert np.isclose(np.linalg.norm(wheel.axis), 1.0)
    assert np.allclose(wheel.axis, [0.0, 0.0, 1.0])


def test_zero_axis_is_rejected():
    with pytest.raises(ValueError):
        ReactionWheel(
            inertia=1e-4, max_torque=1e-3, max_speed=800.0,
            axis=np.array([0.0, 0.0, 0.0]),
        )


def test_max_momentum_matches_definition():
    wheel = make_wheel()
    expected = wheel.inertia * wheel.max_speed
    assert np.isclose(wheel.max_momentum, expected)


# --- Actual torque with clipping --------------------------------------

def test_torque_below_limit_is_delivered_as_is():
    wheel = make_wheel()
    tau_cmd = 0.5e-3     # below wheel.max_torque = 1e-3
    tau = wheel_actual_torque(wheel, wheel_speed=100.0, tau_cmd=tau_cmd)
    assert np.isclose(tau, tau_cmd)


def test_torque_above_limit_is_clipped_to_motor_max():
    wheel = make_wheel()
    tau = wheel_actual_torque(wheel, wheel_speed=0.0, tau_cmd=5.0e-3)
    assert np.isclose(tau, wheel.max_torque)


def test_negative_torque_above_limit_is_clipped_to_minus_motor_max():
    wheel = make_wheel()
    tau = wheel_actual_torque(wheel, wheel_speed=0.0, tau_cmd=-5.0e-3)
    assert np.isclose(tau, -wheel.max_torque)


# --- Speed-based saturation -------------------------------------------

def test_saturated_wheel_cannot_accelerate_further_in_same_direction():
    """At max speed, positive commanded torque delivers zero."""
    wheel = make_wheel()
    tau = wheel_actual_torque(
        wheel, wheel_speed=wheel.max_speed, tau_cmd=+1.0e-3,
    )
    assert tau == 0.0


def test_saturated_wheel_can_still_decelerate():
    """At max speed, negative commanded torque is still delivered."""
    wheel = make_wheel()
    tau = wheel_actual_torque(
        wheel, wheel_speed=wheel.max_speed, tau_cmd=-1.0e-3,
    )
    assert np.isclose(tau, -wheel.max_torque)


def test_symmetric_saturation_in_negative_direction():
    wheel = make_wheel()
    tau = wheel_actual_torque(
        wheel, wheel_speed=-wheel.max_speed, tau_cmd=-1.0e-3,
    )
    assert tau == 0.0


# --- Torque on the spacecraft (Newton III) ----------------------------

def test_wheel_torque_on_spacecraft_is_opposite_and_along_axis():
    """Body-frame reaction torque = -tau_actual * axis."""
    wheel = make_wheel(axis=(0.0, 0.0, 1.0))
    tau_cmd = 0.4e-3
    tau_sc = wheel_torque_on_spacecraft(wheel, wheel_speed=0.0, tau_cmd=tau_cmd)
    assert np.allclose(tau_sc, [0.0, 0.0, -tau_cmd])


def test_wheel_axis_tilted_gives_projected_torque():
    """Torque on spacecraft points along -axis, whatever the axis is."""
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    wheel = ReactionWheel(
        inertia=1e-4, max_torque=1e-3, max_speed=800.0, axis=axis,
    )
    tau_cmd = 0.6e-3
    tau_sc = wheel_torque_on_spacecraft(wheel, wheel_speed=0.0, tau_cmd=tau_cmd)
    assert np.allclose(tau_sc, -tau_cmd * axis)


def test_saturated_wheel_applies_zero_torque_on_spacecraft():
    wheel = make_wheel()
    tau_sc = wheel_torque_on_spacecraft(
        wheel, wheel_speed=wheel.max_speed, tau_cmd=+1.0e-3,
    )
    assert np.allclose(tau_sc, [0.0, 0.0, 0.0])
