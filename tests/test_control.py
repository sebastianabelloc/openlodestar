"""Tests for openlodestar.control.

Run with:
    python -m pytest
"""

import numpy as np

from openlodestar.actuators import ReactionWheel
from openlodestar.control import (
    allocate_to_orthogonal_wheels,
    attitude_error_quaternion,
    pd_torque_on_body,
)
from openlodestar.quaternions import axis_angle_to_quat


Q_ID = np.array([1.0, 0.0, 0.0, 0.0])


# --- Attitude error --------------------------------------------------

def test_error_is_identity_when_current_matches_target():
    q = axis_angle_to_quat(axis=[0, 1, 0], angle_rad=np.deg2rad(37))
    q_e = attitude_error_quaternion(q, q)
    assert np.allclose(q_e, Q_ID)


def test_error_selects_short_rotation():
    """q and -q are the same physical attitude; the error must always
    be returned with a non-negative scalar part."""
    q_current = Q_ID
    # Target is q, but expressed with negative scalar (double cover).
    q_target = -axis_angle_to_quat(axis=[0, 0, 1], angle_rad=np.deg2rad(30))
    q_e = attitude_error_quaternion(q_current, q_target)
    assert q_e[0] >= 0.0
    # The short rotation is 30 deg around +z: q_e_vec should be positive z
    assert q_e[3] > 0.0


def test_error_has_unit_norm():
    q_current = axis_angle_to_quat(axis=[1, 1, 0], angle_rad=np.deg2rad(50))
    q_target = axis_angle_to_quat(axis=[0, 1, 1], angle_rad=np.deg2rad(75))
    q_e = attitude_error_quaternion(q_current, q_target)
    assert np.isclose(np.linalg.norm(q_e), 1.0)


# --- PD control law --------------------------------------------------

def test_pd_returns_zero_torque_when_at_target_and_no_motion():
    q = axis_angle_to_quat(axis=[0, 0, 1], angle_rad=np.deg2rad(45))
    tau = pd_torque_on_body(q_current=q, omega=np.zeros(3),
                            q_target=q, kp=1.0, kd=1.0)
    assert np.allclose(tau, np.zeros(3))


def test_pd_proportional_pulls_toward_target():
    """From identity, target = 60 deg around +z, no motion.
    The controller must command a torque with positive z component."""
    q_current = Q_ID
    q_target = axis_angle_to_quat(axis=[0, 0, 1], angle_rad=np.deg2rad(60))
    tau = pd_torque_on_body(q_current, np.zeros(3), q_target,
                            kp=1.0, kd=0.5)
    # Torque along +z, no cross-axis components
    assert tau[2] > 0.0
    assert np.isclose(tau[0], 0.0)
    assert np.isclose(tau[1], 0.0)


def test_pd_derivative_damps_motion_against_omega():
    """At target already but rotating: torque must oppose omega."""
    q = axis_angle_to_quat(axis=[0, 1, 0], angle_rad=np.deg2rad(20))
    omega = np.array([0.0, 0.5, 0.0])
    tau = pd_torque_on_body(q_current=q, omega=omega, q_target=q,
                            kp=1.0, kd=2.0)
    # Damping: tau in -y direction
    assert np.isclose(tau[0], 0.0)
    assert tau[1] < 0.0
    assert np.isclose(tau[2], 0.0)


def test_pd_gains_can_be_per_axis():
    """kp and kd can be length-3 arrays for anisotropic gains."""
    q_current = Q_ID
    q_target = axis_angle_to_quat(axis=[1, 1, 1], angle_rad=np.deg2rad(30))
    tau_iso = pd_torque_on_body(q_current, np.zeros(3), q_target,
                                kp=2.0, kd=1.0)
    tau_aniso = pd_torque_on_body(q_current, np.zeros(3), q_target,
                                  kp=np.array([2.0, 2.0, 2.0]), kd=1.0)
    assert np.allclose(tau_iso, tau_aniso)


# --- Wheel allocation ------------------------------------------------

def make_orthogonal_wheels():
    return [
        ReactionWheel(inertia=1e-4, max_torque=5e-3, max_speed=800.0, axis=axis)
        for axis in np.eye(3)
    ]


def test_allocation_negates_body_torque_for_orthogonal_wheels():
    wheels = make_orthogonal_wheels()
    body_torque = np.array([0.001, -0.002, 0.0005])
    cmds = allocate_to_orthogonal_wheels(body_torque, wheels)
    assert np.allclose(cmds, -body_torque)


def test_allocation_projects_onto_each_wheel_axis():
    """With wheels at rotated axes, allocation must project properly."""
    axis1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    axis2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    axis3 = np.array([0.0, 0.0, 1.0])
    wheels = [
        ReactionWheel(inertia=1e-4, max_torque=1e-3, max_speed=800.0, axis=a)
        for a in (axis1, axis2, axis3)
    ]
    body_torque = np.array([0.001, 0.0, 0.0])
    cmds = allocate_to_orthogonal_wheels(body_torque, wheels)
    # Each cmd = -body_torque . axis
    expected = np.array([
        -np.dot(body_torque, axis1),
        -np.dot(body_torque, axis2),
        -np.dot(body_torque, axis3),
    ])
    assert np.allclose(cmds, expected)
