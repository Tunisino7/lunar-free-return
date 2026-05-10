"""Numerical propagation with a fourth-order Runge-Kutta integrator."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lunar_free_return.bodies import MassiveBody, SimulationHistory, StateVector
from lunar_free_return.physics import acceleration

Derivative = Callable[[float, StateVector, Sequence[MassiveBody]], StateVector]


def rk4_step(
    time: float,
    step: float,
    state: StateVector,
    derivative: Derivative,
    bodies: Sequence[MassiveBody],
) -> StateVector:
    """Advance a state vector by one fourth-order Runge-Kutta step.

    Parameters
    ----------
    time:
        Current simulation time in seconds.
    step:
        Integration step size in seconds.
    state:
        Current state vector ``[x, y, vx, vy]`` in meters and meters per second.
    derivative:
        Function that maps ``(time, state, bodies)`` to the state derivative.
    bodies:
        Massive bodies passed through to ``derivative``.

    Returns
    -------
    StateVector
        State vector advanced by ``step`` seconds.
    """
    half_step = step / 2.0
    k1 = derivative(time, state, bodies)
    k2 = derivative(time + half_step, state + k1 * half_step, bodies)
    k3 = derivative(time + half_step, state + k2 * half_step, bodies)
    k4 = derivative(time + step, state + k3 * step, bodies)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _collision_body(
    state: StateVector,
    time: float,
    bodies: Sequence[MassiveBody],
) -> MassiveBody | None:
    """Find the first body whose physical radius contains the probe.

    Parameters
    ----------
    state:
        Probe state vector ``[x, y, vx, vy]`` in meters and meters per second.
    time:
        Current simulation time in seconds.
    bodies:
        Bodies to test for intersection.

    Returns
    -------
    MassiveBody | None
        First intersected body, or ``None`` if the probe is outside all body
        radii.
    """
    for body in bodies:
        dx = state[0] - body.position_x(time)
        dy = state[1] - body.position_y(time)
        if dx * dx + dy * dy <= body.radius * body.radius:
            return body
    return None


def propagate_trajectory(
    initial_state: StateVector,
    bodies: Sequence[MassiveBody],
    duration: float,
    time_step: float,
    stop_on_collision: bool = True,
) -> SimulationHistory:
    """Propagate a probe under point-mass gravity.

    Parameters
    ----------
    initial_state:
        Initial probe state ``[x, y, vx, vy]`` in meters and meters per second.
    bodies:
        Massive bodies included in the gravitational model and collision checks.
    duration:
        Maximum propagation time in seconds.
    time_step:
        RK4 integration step in seconds. The final step is shortened when needed
        so the simulation lands exactly on ``duration``.
    stop_on_collision:
        When ``True``, stop as soon as the probe intersects a body's physical
        radius. When ``False``, keep propagating and store the latest detected
        collision body.

    Returns
    -------
    SimulationHistory
        Time samples, state samples, and optional collision body.
    """
    state = np.asarray(initial_state, dtype=float)
    max_steps = int(duration / time_step) + 2
    times = np.empty(max_steps, dtype=float)
    states = np.empty((max_steps, 4), dtype=float)

    time = 0.0
    index = 0
    collision = None
    times[index] = time
    states[index] = state

    while time < duration:
        effective_step = min(float(time_step), float(duration - time))
        state = rk4_step(time, effective_step, state, acceleration, bodies)
        time += effective_step
        index += 1
        times[index] = time
        states[index] = state

        collision = _collision_body(state, time, bodies)
        if collision is not None and stop_on_collision:
            break

    return SimulationHistory(
        times=times[: index + 1],
        states=states[: index + 1],
        collision=collision,
    )
