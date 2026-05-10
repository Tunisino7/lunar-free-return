"""Orbital mechanics helpers for the Earth-Moon free-return model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from lunar_free_return.bodies import (
    NUMBA_AVAILABLE,
    MassiveBody,
    SimulationHistory,
    StateVector,
)
from lunar_free_return.constants import (
    EARTH,
    MOON_ANGULAR_RATE,
    MOON_MASS,
    MOON_ORBIT_RADIUS,
    MOON_RADIUS,
    G,
)

if NUMBA_AVAILABLE:  # pragma: no cover - exercised in the optional accel job.
    from numba import njit
else:  # pragma: no cover - default lightweight install path.
    njit = None

_G = G


def _optional_njit(function):
    """Apply ``numba.njit`` when the optional acceleration extra is installed.

    Parameters
    ----------
    function:
        Function to compile when Numba is importable.

    Returns
    -------
    Callable
        The compiled dispatcher when Numba is available, otherwise the original
        function unchanged. Keeping this as a no-op fallback lets the numerical
        implementation stay single-source.
    """
    if NUMBA_AVAILABLE:
        return njit(cache=True)(function)
    return function


@_optional_njit
def acceleration(
    time: float,
    state: StateVector,
    bodies: Sequence[MassiveBody],  # noqa: UP006 - Numba needs concrete runtime type.
):
    """Compute the state derivative in an N-body gravity field.

    This function is the only acceleration implementation used by the
    propagator. In an accelerated install it is JIT-compiled by
    ``_optional_njit``; in a lightweight install the exact same function body
    runs as regular Python.

    Parameters
    ----------
    time:
        Elapsed simulation time in seconds.
    state:
        Probe state vector ``[x, y, vx, vy]`` in meters and meters per second.
    bodies:
        Massive bodies exerting gravity on the probe.

    Returns
    -------
    StateVector
        Derivative vector ``[vx, vy, ax, ay]`` in meters per second and meters
        per second squared.
    """
    x_probe, y_probe, vx, vy = state
    ax = 0.0
    ay = 0.0
    for body in bodies:
        dx = x_probe - body.position_x(time)
        dy = y_probe - body.position_y(time)
        distance_squared = dx * dx + dy * dy
        inverse_distance_cubed = 1.0 / (distance_squared * math.sqrt(distance_squared))
        ax -= _G * body.mass * dx * inverse_distance_cubed
        ay -= _G * body.mass * dy * inverse_distance_cubed
    return np.array([vx, vy, ax, ay])


def circular_speed(radius: float, central_mass: float) -> float:
    """Compute circular orbit speed around a central body.

    Parameters
    ----------
    radius:
        Orbital radius from the central body center in meters.
    central_mass:
        Central body mass in kilograms.

    Returns
    -------
    float
        Circular orbit speed in meters per second.
    """
    return float(np.sqrt(G * central_mass / radius))


def specific_mechanical_energy(
    state: StateVector,
    bodies: Sequence[MassiveBody],
    time: float,
) -> float:
    """Compute the probe's specific mechanical energy.

    Parameters
    ----------
    state:
        Probe state vector ``[x, y, vx, vy]`` in meters and meters per second.
    bodies:
        Massive bodies contributing gravitational potential energy.
    time:
        Elapsed simulation time in seconds.

    Returns
    -------
    float
        Specific mechanical energy in joules per kilogram.
    """
    probe_position = state[:2]
    probe_velocity = state[2:]
    kinetic = 0.5 * float(np.dot(probe_velocity, probe_velocity))

    potential = 0.0
    for body in bodies:
        distance = float(np.linalg.norm(probe_position - body.position(time)))
        potential -= G * body.mass / distance
    return kinetic + potential


def injection_speed(perigee_radius: float, apogee_radius: float) -> float:
    """Compute Hohmann transfer speed at perigee.

    Parameters
    ----------
    perigee_radius:
        Transfer ellipse perigee radius from Earth's center in meters.
    apogee_radius:
        Transfer ellipse apogee radius from Earth's center in meters.

    Returns
    -------
    float
        Perigee speed in meters per second from the vis-viva equation.
    """
    mu = G * EARTH.mass
    semi_major_axis = 0.5 * (perigee_radius + apogee_radius)
    return float(np.sqrt(mu * (2.0 / perigee_radius - 1.0 / semi_major_axis)))


def phased_moon(initial_phase: float) -> MassiveBody:
    """Create a Moon body with a selected initial orbital phase.

    Parameters
    ----------
    initial_phase:
        Moon phase angle at ``t = 0`` in radians.

    Returns
    -------
    MassiveBody
        Circular-orbit Moon model using the supplied phase.
    """
    return MassiveBody(
        MOON_MASS,
        MOON_RADIUS,
        0.0,
        0.0,
        MOON_ANGULAR_RATE,
        MOON_ORBIT_RADIUS,
        initial_phase,
        "Moon",
    )


def lunar_closest_approach(
    history: SimulationHistory,
    moon: MassiveBody,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Find the closest approach between the probe and the Moon.

    Parameters
    ----------
    history:
        Propagated probe state history.
    moon:
        Moon body used to compute lunar positions over ``history.times``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, float, float]
        Moon positions in meters, probe-Moon center distances in meters, the
        closest-approach index, the closest distance in meters, and the
        closest-approach time in seconds.
    """
    probe_positions = history.states[:, :2]
    moon_positions = np.array([moon.position(t) for t in history.times], dtype=float)
    distances = np.linalg.norm(probe_positions - moon_positions, axis=1)
    closest_index = int(np.argmin(distances))
    return (
        moon_positions,
        distances,
        closest_index,
        float(distances[closest_index]),
        float(history.times[closest_index]),
    )
