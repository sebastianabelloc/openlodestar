# OpenLodestar

Spacecraft attitude determination and control (ADCS) simulator in Python. This is also a learning project:
the author is learning GNC by building it, so understanding matters more than speed.

## How to work here

- The math of dynamics, control and estimation is written by the author. When a task touches equations,
  control laws or filters, explain the idea and leave the core lines to him (TODO(human)), then review what he writes.
- Ask before large changes (new modules, refactors, changing public function names or signatures).
  Propose a plan first.
- Do not edit or remove existing comments, docstrings or author signatures unless asked.
- Keep the existing style: plain lowercase `# ...` comments, NumPy-style docstrings, no fancy alignment.
- Every change must keep the test suite green. Add or update tests for any new behaviour.

## Conventions

- Quaternions: Hamilton, scalar first `(w, x, y, z)`, mapping body vectors to inertial vectors.
- Angular velocity in the body frame. Inertia as a diagonal `(Ixx, Iyy, Izz)` for now.
- SI units everywhere (kg, m, s, rad, N m).
- Integration with `scipy.integrate.solve_ivp`.

## Commands

```bash
pip install -e ".[dev]"     # install
python -m pytest -q         # tests (must pass before every commit)
python examples/slew_and_hold.py
```

## Layout

- `openlodestar/dynamics.py`: Euler's equations, spacecraft + wheels dynamics
- `openlodestar/actuators.py`: reaction wheel model
- `openlodestar/control.py`: error quaternion, PD law, wheel allocation
- `openlodestar/utils/quaternions.py`: quaternion operations
- `examples/`, `tests/`

## Before publishing

Before updating the README, a release or anything public, use the `reviewer` subagent to check that every number
and equation in the text matches what the code actually does.
