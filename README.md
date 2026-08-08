# OpenLodestar

Open-source **spacecraft Attitude Determination and Control (ADCS)** simulator, written in Python.

A *lodestar* is a navigation star. OpenLodestar simulates the full closed loop a satellite needs to go from tumbling to precisely pointing at a target and holding that orientation against disturbances.

> **Status:** early development — Phase 0 (foundations). This README describes the project goal; features marked *planned* are not implemented yet.

---

## What it will do (v1.0)

A CubeSat starts with an arbitrary initial attitude and body rate, a controller slews it to a commanded attitude and holds it against a disturbance torque, while an estimator recovers the true attitude from noisy sensors.

The closed loop being simulated:

**Dynamics** (truth) → **Sensors** (noisy measurements) → **Estimator** (attitude estimate) → **Controller** (commanded torque) → **Actuators** (real torque with saturation) → *feeds back to Dynamics*.

## v1.0 scope

- Rigid-body attitude dynamics (quaternions, Euler's equations).
- Reaction wheels with saturation (3 orthogonal or 4 in a pyramid).
- Controller that slews to a commanded attitude and holds against disturbance.
- Estimator that recovers attitude from a noisy gyro + star tracker (TRIAD/QUEST + MEKF).
- Basic 3D visualization and telemetry.
- Demo: *point and hold* on a 3U CubeSat.

## Installation

*Not applicable yet — the package is not published. Instructions will land here as soon as there is runnable code.*

Planned requirements: Python 3.11+, NumPy, SciPy, Matplotlib, pytest.

## Intended usage

A typical user will define a spacecraft, pick a scenario, run the simulation, and visualize the result. The target session is ~15 lines of Python. Sensible defaults, everything overridable.

## Target repository layout

```
openlodestar/
├── README.md
├── LICENSE                       (MIT)
├── requirements.txt
├── src/openlodestar/
│   ├── spacecraft.py             spacecraft: inertia, state
│   ├── dynamics.py               Euler's equations, quaternion kinematics
│   ├── actuators.py              reaction wheels, saturation
│   ├── sensors.py                gyro, star tracker + noise
│   ├── estimator.py              TRIAD/QUEST, MEKF
│   ├── control.py                quaternion PD control
│   ├── simulation.py             simulation loop, integration
│   ├── visualization.py          3D + telemetry
│   └── utils/quaternions.py      quaternion operations
├── examples/
│   └── cubesat_point_and_hold.py
├── tests/
└── docs/
```

## Technical conventions

- **Quaternions:** single consistent convention (Hamilton). Documented in `utils/quaternions.py`.
- **Frames:** inertial (ECI) and body.
- **Units:** SI (kg, m, s, rad).
- **Integration:** fixed-step RK4 or `scipy.integrate.solve_ivp` (RK45).
- **Physical validation:** angular momentum and energy conservation used as dynamics tests.

## Roadmap by phase

- **Phase 0 — Visible foundations.** Reference frames, rotation matrices, quaternions, ODE integration. Deliverable: rotate a 3D cube with a rotation matrix and with a quaternion.
- **Phase 1 — Dynamics.** Free rigid body: quaternion kinematics, inertia tensor, Euler's equations. Deliverable: tumbling body in 3D with telemetry.
- **Phase 2 — Actuators + Control.** Reaction wheels with saturation, quaternion PD control. Deliverable: slew to a commanded attitude and hold against disturbance.
- **Phase 3 — Sensors + Estimator.** Noisy gyro and star tracker, TRIAD/QUEST and MEKF. Deliverable: realistic closed loop using the estimated attitude.
- **Phase 4 — Mission + C++ + validation.** Detumble → point → hold; core port to C++; cross-check against MATLAB/Simulink.

Beyond v1.0: YAML configuration, magnetorquers for wheel desaturation, CMGs, richer visualization.

## Theoretical references

- Wertz, J. R. — *Spacecraft Attitude Determination and Control*.
- Markley, F. L. & Crassidis, J. L. — *Fundamentals of Spacecraft Attitude Determination and Control*.
- Sidi, M. J. — *Spacecraft Dynamics and Control: A Practical Engineering Approach*.
- 3Blue1Brown — *Essence of Linear Algebra* (linear-algebra background).

## License

MIT — see [`LICENSE`](LICENSE). Anyone may use, modify, and commercialize the code, keeping the copyright notice.

## Author

Sebastián Abello — aerospace engineer focused on GNC. OpenLodestar is both a publishable tool and the vehicle I'm using to learn GNC by building it.
