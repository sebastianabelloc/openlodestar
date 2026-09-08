"""Tests for openlodestar.dynamics.

Run from the repo root with:
    python -m pytest
"""

import numpy as np

from openlodestar.dynamics import angular_acceleration, state_derivative


# --- Sample inertia tensors (arbitrary but physically plausible) ------
# Slightly non-symmetric CubeSat-scale inertia in kg*m^2.
I_GENERIC = np.array([0.10, 0.12, 0.15])
# Axisymmetric body: Ixx == Iyy != Izz. Common approximation.
I_AXISYMMETRIC = np.array([0.10, 0.10, 0.15])


def test_zero_omega_zero_torque_gives_zero_acceleration():
    """Body at rest with no torque stays at rest."""
    omega = np.zeros(3)
    assert np.allclose(angular_acceleration(omega, I_GENERIC), np.zeros(3))


def test_spin_around_principal_axis_is_steady_state():
    """Rotation about a single principal axis, no torque -> no acceleration."""
    for axis_index in range(3):
        omega = np.zeros(3)
        omega[axis_index] = 1.5  # rad/s
        acc = angular_acceleration(omega, I_GENERIC)
        assert np.allclose(acc, np.zeros(3)), f"principal axis {axis_index} not steady"


def test_axisymmetric_spin_around_symmetry_axis_is_steady():
    """For Ixx = Iyy, spin around z is trivially steady even with wobble in xy."""
    omega = np.array([0.0, 0.0, 2.0])
    acc = angular_acceleration(omega, I_AXISYMMETRIC)
    assert np.allclose(acc, np.zeros(3))


def test_off_axis_spin_is_not_steady_for_asymmetric_body():
    """A non-principal-axis rotation on an asymmetric body accelerates."""
    omega = np.array([1.0, 2.0, 3.0])
    acc = angular_acceleration(omega, I_GENERIC)
    assert not np.allclose(acc, np.zeros(3))


def test_kinetic_energy_is_instantaneously_conserved_in_free_body():
    """d/dt (0.5 * omega . I . omega) = omega . I . omega_dot = omega . tau.

    With tau = 0, the rate of change of kinetic energy is zero
    instantaneously. Verify omega . (I . omega_dot) == 0 for arbitrary
    omega on an arbitrary asymmetric body.
    """
    rng = np.random.default_rng(seed=0)
    omega = rng.standard_normal(3)
    omega_dot = angular_acceleration(omega, I_GENERIC)
    dT_dt = np.dot(omega, I_GENERIC * omega_dot)
    assert np.isclose(dT_dt, 0.0, atol=1e-12)


def test_angular_momentum_change_matches_applied_torque():
    """dh/dt|body = tau - omega x h, so I . omega_dot + omega x (I omega) = tau."""
    omega = np.array([0.3, -0.4, 0.2])
    tau = np.array([0.01, 0.0, -0.005])
    omega_dot = angular_acceleration(omega, I_GENERIC, torque=tau)

    h = I_GENERIC * omega
    reconstructed_tau = I_GENERIC * omega_dot + np.cross(omega, h)
    assert np.allclose(reconstructed_tau, tau)


# --- Tests for the combined state derivative --------------------------

def test_state_derivative_shape_and_split():
    """state_derivative must return a (7,) vector combining q_dot and omega_dot."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.1, 0.2, -0.3])
    y = np.concatenate([q, omega])
    y_dot = state_derivative(0.0, y, I_GENERIC)
    assert y_dot.shape == (7,)


def test_state_derivative_rest_is_stationary():
    """Body at rest with unit-quaternion attitude and no torque: y_dot == 0."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    y = np.concatenate([q, omega])
    y_dot = state_derivative(0.0, y, I_GENERIC)
    assert np.allclose(y_dot, np.zeros(7))


def test_state_derivative_pure_z_spin_matches_analytical():
    """From identity attitude and constant spin around +z, q_dot must match
    the analytical value (0, 0, 0, w_z/2) and omega_dot must be zero."""
    w_z = 0.9
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, w_z])
    y = np.concatenate([q, omega])
    y_dot = state_derivative(0.0, y, I_GENERIC)

    assert np.allclose(y_dot[:4], [0.0, 0.0, 0.0, w_z / 2])
    assert np.allclose(y_dot[4:], np.zeros(3))
