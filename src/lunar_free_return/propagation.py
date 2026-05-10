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
    """Advance ``state`` by one fourth-order Runge-Kutta step."""
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
    """Propagate a probe under the gravity of ``bodies``."""
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
