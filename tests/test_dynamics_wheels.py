"""Tests for the coupled spacecraft-plus-wheels dynamics.

These tests cover ``state_derivative_with_wheels`` and its helpers
(``wheel_momentum_body``, ``total_angular_momentum_body``).

Run with:
    python -m pytest
"""

import numpy as np
from scipy.integrate import solve_ivp

from openlodestar.actuators import ReactionWheel
from openlodestar.dynamics import (
    state_derivative_with_wheels,
    total_angular_momentum_body,
    wheel_momentum_body,
)
from openlodestar.quaternions import rotate_vector


# --- Shared configuration: 3-orthogonal-wheel CubeSat ------------------
INERTIA = np.array([0.10, 0.12, 0.15])
Q_ID = np.array([1.0, 0.0, 0.0, 0.0])


def make_wheels():
    """Three identical wheels aligned with body x, y, z."""
    return [
        ReactionWheel(inertia=1e-4, max_torque=5e-3, max_speed=800.0, axis=axis)
        for axis in np.eye(3)
    ]


def zero_cmds(t, y):
    return np.zeros(3)


# --- Basic properties -------------------------------------------------

def test_state_derivative_shape_with_three_wheels():
    """Return vector must have 7 + n_wheels components."""
    wheels = make_wheels()
    y = np.concatenate([Q_ID, np.zeros(3), np.zeros(3)])
    y_dot = state_derivative_with_wheels(0.0, y, INERTIA, wheels, zero_cmds)
    assert y_dot.shape == (10,)


def test_rest_state_with_stationary_wheels_is_stationary():
    """No motion, no commanded torque, no perturbation -> everything stays."""
    wheels = make_wheels()
    y = np.concatenate([Q_ID, np.zeros(3), np.zeros(3)])
    y_dot = state_derivative_with_wheels(0.0, y, INERTIA, wheels, zero_cmds)
    assert np.allclose(y_dot, np.zeros(10))


def test_wheel_momentum_body_matches_sum_of_projections():
    wheels = make_wheels()
    speeds = np.array([10.0, -20.0, 5.0])
    h = wheel_momentum_body(wheels, speeds)
    expected = np.array([1e-4 * 10.0, 1e-4 * -20.0, 1e-4 * 5.0])
    assert np.allclose(h, expected)


# --- Newton III sanity: commanding one wheel accelerates the body -----

def test_commanding_z_wheel_causes_negative_body_spin_around_z():
    """From rest, positive command on the z-wheel must produce
    omega_dot with a negative z component (Newton III reaction)."""
    wheels = make_wheels()
    y = np.concatenate([Q_ID, np.zeros(3), np.zeros(3)])

    def cmds(t, y):
        return np.array([0.0, 0.0, 1e-3])   # torque on z-wheel

    y_dot = state_derivative_with_wheels(0.0, y, INERTIA, wheels, cmds)
    omega_dot = y_dot[4:7]
    wheel_dot = y_dot[7:]

    # z-wheel accelerates in +z (positive Omega_dot)
    assert wheel_dot[2] > 0.0
    # spacecraft body accelerates in -z direction
    assert omega_dot[2] < 0.0
    # x and y stay untouched
    assert np.isclose(omega_dot[0], 0.0)
    assert np.isclose(omega_dot[1], 0.0)


# --- Saturation propagates end-to-end ---------------------------------

def test_saturated_wheel_stops_producing_body_reaction():
    """If the z-wheel is at max speed and we command more +torque, both
    Omega_z_dot and the induced omega_body_dot around z must be zero."""
    wheels = make_wheels()
    wheel_speeds = np.array([0.0, 0.0, wheels[2].max_speed])
    y = np.concatenate([Q_ID, np.zeros(3), wheel_speeds])

    def cmds(t, y):
        return np.array([0.0, 0.0, 1e-3])

    y_dot = state_derivative_with_wheels(0.0, y, INERTIA, wheels, cmds)
    omega_dot = y_dot[4:7]
    wheel_dot = y_dot[7:]

    # Wheel z can't accelerate any further
    assert np.isclose(wheel_dot[2], 0.0)
    # Consequently the body receives no reaction torque
    assert np.allclose(omega_dot, np.zeros(3))


# --- Integration test: total angular momentum conserved in inertial ---

def test_inertial_angular_momentum_is_conserved_over_short_run():
    """With no external torque, |h_inertial| must be constant even
    when wheels are actively commanded (they only redistribute
    momentum between body and wheels)."""
    wheels = make_wheels()
    q0 = Q_ID
    omega0 = np.array([0.1, -0.2, 0.05])
    wheel_speeds0 = np.array([5.0, -3.0, 8.0])
    y0 = np.concatenate([q0, omega0, wheel_speeds0])

    def cmds(t, y):
        # Small time-varying commanded torque profile
        return np.array([
            2e-4 * np.sin(1.0 * t),
            1e-4 * np.cos(0.5 * t),
            -3e-4 * np.sin(0.7 * t),
        ])

    sol = solve_ivp(
        state_derivative_with_wheels,
        t_span=(0.0, 5.0),
        y0=y0,
        args=(INERTIA, wheels, cmds),
        method="RK45",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success

    t = np.linspace(0.0, 5.0, 200)
    y = sol.sol(t)

    h_inertial_magnitude = np.empty(t.size)
    for k in range(t.size):
        q = y[:4, k]
        omega = y[4:7, k]
        wheel_speeds = y[7:, k]
        h_body = total_angular_momentum_body(INERTIA, omega, wheels, wheel_speeds)
        h_inertial = rotate_vector(q, h_body)
        h_inertial_magnitude[k] = np.linalg.norm(h_inertial)

    relative_variation = (
        (h_inertial_magnitude.max() - h_inertial_magnitude.min())
        / h_inertial_magnitude[0]
    )
    assert relative_variation < 1e-7, (
        f"|h_inertial| varied by {relative_variation:.2e} (relative)"
    )
