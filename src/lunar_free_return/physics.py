"""Orbital mechanics helpers for the Earth-Moon free-return model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lunar_free_return.bodies import MassiveBody, SimulationHistory, StateVector
from lunar_free_return.constants import (
    EARTH,
    G,
    MOON_ANGULAR_RATE,
    MOON_MASS,
    MOON_ORBIT_RADIUS,
    MOON_RADIUS,
)


def acceleration(
    time: float,
    state: StateVector,
    bodies: Sequence[MassiveBody],
) -> StateVector:
    """Return ``[vx, vy, ax, ay]`` for a probe in an N-body gravity field."""
    x_probe, y_probe, vx, vy = state
    ax = 0.0
    ay = 0.0
    for body in bodies:
        dx = x_probe - body.position_x(time)
        dy = y_probe - body.position_y(time)
        distance_squared = dx * dx + dy * dy
        inverse_distance_cubed = 1.0 / (distance_squared * np.sqrt(distance_squared))
        ax -= G * body.mass * dx * inverse_distance_cubed
        ay -= G * body.mass * dy * inverse_distance_cubed
    return np.array([vx, vy, ax, ay], dtype=float)


def circular_speed(radius: float, central_mass: float) -> float:
    """Return circular orbit speed in m/s."""
    return float(np.sqrt(G * central_mass / radius))


def specific_mechanical_energy(
    state: StateVector,
    bodies: Sequence[MassiveBody],
    time: float,
) -> float:
    """Return specific mechanical energy in J/kg."""
    probe_position = state[:2]
    probe_velocity = state[2:]
    kinetic = 0.5 * float(np.dot(probe_velocity, probe_velocity))

    potential = 0.0
    for body in bodies:
        distance = float(np.linalg.norm(probe_position - body.position(time)))
        potential -= G * body.mass / distance
    return kinetic + potential


def injection_speed(perigee_radius: float, apogee_radius: float) -> float:
    """Return Hohmann transfer perigee speed from the vis-viva equation."""
    mu = G * EARTH.mass
    semi_major_axis = 0.5 * (perigee_radius + apogee_radius)
    return float(np.sqrt(mu * (2.0 / perigee_radius - 1.0 / semi_major_axis)))


def phased_moon(initial_phase: float) -> MassiveBody:
    """Return the Moon body with a selected initial orbital phase."""
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
    """Return Moon positions, probe-Moon distances, index, distance, and time."""
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
