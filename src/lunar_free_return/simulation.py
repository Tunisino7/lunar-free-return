"""Simulation of Schwaniger Earth-Moon free-return trajectories."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from lunar_free_return.bodies import MassiveBody, StateVector
from lunar_free_return.constants import (
    EARTH,
    EARTH_RADIUS,
    G,
    MOON_ANGULAR_RATE,
    MOON_ORBIT_RADIUS,
    MOON_RADIUS,
)
from lunar_free_return.physics import injection_speed, lunar_closest_approach, phased_moon
from lunar_free_return.propagation import propagate_trajectory
from lunar_free_return.types import (
    OrbitalDirection,
    PRESETS,
    ReturnType,
    SimulationConfig,
    SimulationResult,
    TrajectoryCase,
)


def preset(case: TrajectoryCase | str) -> SimulationConfig:
    """Return a configured preset for one Schwaniger case."""
    return PRESETS[TrajectoryCase(case)]


def classify_return(
    min_moon_distance: float,
    max_earth_distance: float,
    return_detected: bool,
    collision_name: str | None,
) -> tuple[ReturnType, str]:
    """Classify the trajectory from distance and collision indicators."""
    if collision_name == "Moon":
        return (
            ReturnType.LUNAR_IMPACT,
            "Lunar impact: the probe reached the Moon's surface.",
        )

    lunar_zone_threshold = 0.75 * MOON_ORBIT_RADIUS
    reached_lunar_zone = max_earth_distance >= lunar_zone_threshold

    if return_detected and reached_lunar_zone:
        altitude_km = (min_moon_distance - MOON_RADIUS) / 1e3
        return (
            ReturnType.FREE_RETURN,
            (
                f"Free return confirmed: apogee {max_earth_distance / 1e6:.0f} Mm, "
                f"lunar flyby altitude {altitude_km:,.0f} km, ballistic Earth return."
            ),
        )

    if return_detected and not reached_lunar_zone:
        return (
            ReturnType.DIRECT_RETURN,
            (
                "Direct return: the probe returned without reaching the lunar zone "
                f"(apogee {max_earth_distance / 1e3:,.0f} km, "
                f"threshold {lunar_zone_threshold / 1e3:,.0f} km). "
                "Increase the speed factor to target the Moon."
            ),
        )

    if not return_detected and reached_lunar_zone:
        return (
            ReturnType.FLYBY_NO_RETURN,
            (
                "Flyby without detected return: the probe reached the lunar zone but "
                "did not return to Earth within the simulated time window."
            ),
        )

    return (
        ReturnType.ESCAPED,
        "No Earth return was detected within the simulated time window.",
    )


def build_initial_conditions(
    config: SimulationConfig,
) -> tuple[StateVector, MassiveBody, float]:
    """Build the probe state and phased Moon for a free-return trajectory."""
    perigee_radius = EARTH_RADIUS + config.departure_altitude
    apogee_radius = MOON_ORBIT_RADIUS
    initial_speed = injection_speed(perigee_radius, apogee_radius) * config.speed_factor

    semi_major_axis = 0.5 * (perigee_radius + apogee_radius)
    transfer_time = float(np.pi * np.sqrt(semi_major_axis**3 / (G * EARTH.mass)))
    moon_phase = -MOON_ANGULAR_RATE * transfer_time + config.moon_phase_adjustment
    moon = phased_moon(moon_phase)

    velocity_sign = float(config.orbital_direction)
    initial_state = np.array(
        [-perigee_radius, 0.0, 0.0, -initial_speed * velocity_sign],
        dtype=float,
    )
    return initial_state, moon, initial_speed


def simulate(config: SimulationConfig | None = None) -> SimulationResult:
    """Run a free-return trajectory simulation."""
    config = config or SimulationConfig()
    initial_state, moon, initial_speed = build_initial_conditions(config)

    history = propagate_trajectory(
        initial_state=initial_state,
        bodies=[EARTH, moon],
        duration=config.duration,
        time_step=config.time_step,
        stop_on_collision=True,
    )

    probe_positions = history.states[:, :2]
    earth_distances = np.linalg.norm(probe_positions, axis=1)
    _, _, _, min_moon_distance, moon_closest_time = lunar_closest_approach(
        history, moon
    )

    apogee_index = int(np.argmax(earth_distances))
    max_earth_distance = float(earth_distances[apogee_index])
    return_threshold = (config.return_altitude_threshold_km + EARTH_RADIUS / 1e3) * 1e3

    return_detected = False
    earth_return_time = None
    earth_return_distance = None

    if apogee_index < len(earth_distances) - 1:
        post_apogee_distances = earth_distances[apogee_index:]
        if np.min(post_apogee_distances) < return_threshold:
            return_detected = True
            relative_return_index = int(np.argmin(post_apogee_distances))
            return_index = apogee_index + relative_return_index
            earth_return_time = float(history.times[return_index])
            earth_return_distance = float(earth_distances[return_index])

    collision_name = history.collision.name if history.collision is not None else None
    if collision_name == "Earth" and not return_detected:
        return_detected = True
        earth_return_time = float(history.times[-1])
        earth_return_distance = float(earth_distances[-1])

    return_type, diagnostic = classify_return(
        min_moon_distance=min_moon_distance,
        max_earth_distance=max_earth_distance,
        return_detected=return_detected,
        collision_name=collision_name,
    )

    return SimulationResult(
        history=history,
        moon=moon,
        apogee_index=apogee_index,
        min_moon_distance=min_moon_distance,
        max_earth_distance=max_earth_distance,
        earth_return_distance=earth_return_distance,
        moon_closest_approach_time=float(moon_closest_time),
        earth_return_time=earth_return_time,
        total_duration=float(history.times[-1]),
        injection_speed=initial_speed,
        return_type=return_type,
        diagnostic=diagnostic,
        case=config.case,
    )


def with_case(case: TrajectoryCase | str, **overrides: object) -> SimulationConfig:
    """Return a preset config with selected dataclass field overrides."""
    return replace(preset(case), **overrides)
