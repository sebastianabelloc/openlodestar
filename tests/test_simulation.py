"""Integration tests: verify that a full ODE simulation respects the
conservation laws of a torque-free rigid body.

These tests actually integrate the coupled attitude/angular-velocity
system with ``scipy.integrate.solve_ivp`` and check that:

  - the quaternion norm stays close to 1,
  - the angular momentum magnitude |h| is (nearly) constant,
  - the rotational kinetic energy T is (nearly) constant.

If any of the three drift beyond tolerance, either the physics
implementation or the integrator setup is wrong.

Run with:
    python -m pytest
"""

import numpy as np
from scipy.integrate import solve_ivp

from openlodestar.dynamics import state_derivative


# --- Shared setup: asymmetric CubeSat, initial tumble ------------------
INERTIA = np.array([0.10, 0.12, 0.15])
Q0 = np.array([1.0, 0.0, 0.0, 0.0])
OMEGA0 = np.array([0.4, 0.9, 0.2])
Y0 = np.concatenate([Q0, OMEGA0])

T_END = 5.0
N_SAMPLES = 200


def _run_free_body_simulation():
    """Integrate the free-body system and return sampled state components."""
    sol = solve_ivp(
        state_derivative,
        t_span=(0.0, T_END),
        y0=Y0,
        args=(INERTIA,),
        method="RK45",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success, f"Integration failed: {sol.message}"

    t = np.linspace(0.0, T_END, N_SAMPLES)
    y = sol.sol(t)
    q = y[:4]           # shape (4, N_SAMPLES)
    omega = y[4:]       # shape (3, N_SAMPLES)
    return q, omega


def test_free_body_quaternion_norm_stays_unit():
    """|q| must remain 1 within a small tolerance during a free-body run."""
    q, _ = _run_free_body_simulation()
    norms = np.linalg.norm(q, axis=0)
    max_deviation = np.max(np.abs(norms - 1.0))
    assert max_deviation < 1e-7, f"quaternion norm drifted by {max_deviation}"


def test_free_body_angular_momentum_is_conserved():
    """|h| = |I . omega| must stay constant when tau = 0."""
    _, omega = _run_free_body_simulation()
    h_body = INERTIA[:, None] * omega
    h_magnitude = np.linalg.norm(h_body, axis=0)

    relative_variation = (h_magnitude.max() - h_magnitude.min()) / h_magnitude[0]
    assert relative_variation < 1e-7, (
        f"|h| varied by {relative_variation:.2e} (relative)"
    )


def test_free_body_kinetic_energy_is_conserved():
    """T = 0.5 * omega . (I . omega) must stay constant when tau = 0."""
    _, omega = _run_free_body_simulation()
    kinetic = 0.5 * np.sum(INERTIA[:, None] * omega ** 2, axis=0)

    relative_variation = (kinetic.max() - kinetic.min()) / kinetic[0]
    assert relative_variation < 1e-7, (
        f"T varied by {relative_variation:.2e} (relative)"
    )
