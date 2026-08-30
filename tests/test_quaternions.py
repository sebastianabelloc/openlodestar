"""Basic tests for openlodestar.utils.quaternions.

Run from the repo root with:
    pytest
"""

import numpy as np
import pytest

from openlodestar.utils.quaternions import (
    axis_angle_to_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_vector,
)


# --- Constants used across tests --------------------------------------
Q_ID = np.array([1.0, 0.0, 0.0, 0.0])   # identity quaternion


def test_axis_angle_gives_unit_quaternion():
    """Any quaternion built from axis-angle must have norm 1."""
    q = axis_angle_to_quat(axis=[1, 1, 1], angle_rad=np.deg2rad(120))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_identity_multiplication_is_neutral():
    """q_id * q == q * q_id == q for any q."""
    q = axis_angle_to_quat(axis=[0, 1, 0], angle_rad=np.deg2rad(37))
    assert np.allclose(quat_mul(Q_ID, q), q)
    assert np.allclose(quat_mul(q, Q_ID), q)


def test_rotation_axis_is_invariant():
    """Rotating the rotation axis itself must leave it unchanged."""
    axis = np.array([1.0, 0.0, 0.0])
    q = axis_angle_to_quat(axis=axis, angle_rad=np.deg2rad(90))
    assert np.allclose(quat_rotate_vector(q, axis), axis)


def test_90deg_x_rotation_sends_y_to_z():
    """A 90 deg rotation about +x sends +y to +z (right-hand rule)."""
    q = axis_angle_to_quat(axis=[1, 0, 0], angle_rad=np.pi / 2)
    v = np.array([0.0, 1.0, 0.0])
    assert np.allclose(quat_rotate_vector(q, v), [0.0, 0.0, 1.0])


def test_quaternion_matches_rotation_matrix():
    """Rotating a vector with q must match rotating it with the equivalent R."""
    angle = np.pi / 2
    c, s = np.cos(angle), np.sin(angle)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    q = axis_angle_to_quat(axis=[1, 0, 0], angle_rad=angle)
    v = np.array([1.0, 1.0, 0.0])

    v_via_quat = quat_rotate_vector(q, v)
    v_via_matrix = Rx @ v
    assert np.allclose(v_via_quat, v_via_matrix)


def test_quaternions_do_not_commute():
    """q_x * q_y != q_y * q_x in general, mirroring rotation non-commutativity."""
    q_x = axis_angle_to_quat(axis=[1, 0, 0], angle_rad=np.pi / 2)
    q_y = axis_angle_to_quat(axis=[0, 1, 0], angle_rad=np.pi / 2)
    assert not np.allclose(quat_mul(q_x, q_y), quat_mul(q_y, q_x))


def test_conjugate_negates_vector_part():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(quat_conjugate(q), [0.5, -0.5, -0.5, -0.5])
