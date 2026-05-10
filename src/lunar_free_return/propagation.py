"""Numerical propagation with a fourth-order Runge-Kutta integrator."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lunar_free_return.bodies import (
    NUMBA_AVAILABLE,
    MassiveBody,
    SimulationHistory,
    StateVector,
)
from lunar_free_return.physics import acceleration

if NUMBA_AVAILABLE:  # pragma: no cover - exercised in the optional accel job.
    from numba import njit
    from numba.typed import List as TypedList
else:  # pragma: no cover - default lightweight install path.
    njit = None
    TypedList = None

Derivative = Callable[[float, StateVector, Sequence[MassiveBody]], StateVector]


def _optional_njit(function=None, *, cache: bool = True):
    """Decorate a function with ``numba.njit`` when Numba is available.

    Parameters
    ----------
    function:
        Function to decorate. When omitted, the helper returns a configured
        decorator so keyword arguments such as ``cache=False`` can be used.
    cache:
        Whether Numba should cache the compiled function. The propagation core
        disables caching because it receives a dynamic typed list, while smaller
        numerical helpers can be cached safely.

    Returns
    -------
    Callable
        A Numba dispatcher in accelerated installs, or the original Python
        function in lightweight installs.
    """

    def decorate(inner):
        if NUMBA_AVAILABLE:
            return njit(cache=cache)(inner)
        return inner

    if function is None:
        return decorate
    return decorate(function)


def _body_sequence(bodies: Sequence[MassiveBody]):
    """Return a body container suitable for the active runtime.

    Parameters
    ----------
    bodies:
        Python sequence of massive bodies passed by public callers.

    Returns
    -------
    Sequence[MassiveBody]
        The original sequence in the fallback runtime, or a ``numba.typed.List``
        containing the same bodies when Numba is active.
    """
    if not NUMBA_AVAILABLE:
        return bodies
    typed = TypedList()
    for body in bodies:
        typed.append(body)
    return typed


@_optional_njit
def rk4_step(
    time: float,
    step: float,
    state: StateVector,
    derivative: Derivative,
    bodies: Sequence[MassiveBody],
):
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


@_optional_njit
def _collision_index(
    state: StateVector,
    time: float,
    bodies: Sequence[MassiveBody],
) -> int:
    """Find the first body intersected by the probe.

    Parameters
    ----------
    state:
        Probe state vector ``[x, y, vx, vy]`` in meters and meters per second.
    time:
        Current propagation time in seconds.
    bodies:
        Runtime body sequence to check. This is either a Python sequence or a
        Numba typed list, depending on the active backend.

    Returns
    -------
    int
        Index of the first intersected body, or ``-1`` when no collision is
        detected.
    """
    for index in range(len(bodies)):
        body = bodies[index]
        dx = state[0] - body.position_x(time)
        dy = state[1] - body.position_y(time)
        if dx * dx + dy * dy <= body.radius * body.radius:
            return index
    return -1


@_optional_njit(cache=False)
def _propagate_core(
    initial_state: StateVector,
    bodies: Sequence[MassiveBody],
    duration: float,
    time_step: float,
    stop_on_collision: bool,
):
    """Propagate state history with RK4 and optional collision stopping.

    This is the single integration loop used by both runtimes. The public
    wrapper prepares the body container, then this function runs either as
    compiled Numba code or as normal Python depending on ``_optional_njit``.

    Parameters
    ----------
    initial_state:
        Initial probe state ``[x, y, vx, vy]`` in meters and meters per second.
    bodies:
        Runtime body sequence used for gravity and collision checks.
    duration:
        Maximum propagation duration in seconds.
    time_step:
        RK4 step size in seconds.
    stop_on_collision:
        Whether to stop immediately after a collision is detected.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        Sample times in seconds, state history with rows ``[x, y, vx, vy]``,
        and the collision body index or ``-1``.
    """
    max_steps = int(duration / time_step) + 2
    times = np.empty(max_steps)
    states = np.empty((max_steps, 4))

    time = 0.0
    index = 0
    collision_index = -1
    state = initial_state.copy()
    times[index] = time
    states[index] = state

    while time < duration:
        effective_step = min(time_step, duration - time)
        state = rk4_step(time, effective_step, state, acceleration, bodies)
        time += effective_step
        index += 1
        times[index] = time
        states[index] = state

        collision_index = _collision_index(state, time, bodies)
        if collision_index >= 0 and stop_on_collision:
            break

    return times[: index + 1], states[: index + 1], collision_index


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
    body_sequence = _body_sequence(bodies)
    times, states, collision_index = _propagate_core(
        np.asarray(initial_state, dtype=np.float64),
        body_sequence,
        float(duration),
        float(time_step),
        bool(stop_on_collision),
    )
    return SimulationHistory(
        times=times,
        states=states,
        collision=bodies[collision_index] if collision_index >= 0 else None,
    )
