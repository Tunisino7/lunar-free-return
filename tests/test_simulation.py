from dataclasses import replace

import numpy as np
import pytest

from lunar_free_return import (
    OrbitalDirection,
    ReturnType,
    TrajectoryCase,
    build_initial_conditions,
    simulate,
    with_case,
)
from lunar_free_return.bodies import NUMBA_AVAILABLE
from lunar_free_return.constants import EARTH, EARTH_RADIUS, MOON_ORBIT_RADIUS
from lunar_free_return.propagation import propagate_trajectory
from lunar_free_return.simulation import classify_return


def test_initial_direction_changes_velocity_sign() -> None:
    prograde = with_case("Ai", orbital_direction=OrbitalDirection.PROGRADE)
    retrograde = replace(prograde, orbital_direction=OrbitalDirection.RETROGRADE)

    prograde_state, _, _ = build_initial_conditions(prograde)
    retrograde_state, _, _ = build_initial_conditions(retrograde)

    assert prograde_state[3] < 0
    assert retrograde_state[3] > 0
    assert np.isclose(abs(prograde_state[3]), abs(retrograde_state[3]))


def test_classifies_free_return_when_probe_reaches_lunar_zone_and_returns() -> None:
    return_type, diagnostic = classify_return(
        min_moon_distance=MOON_ORBIT_RADIUS - EARTH_RADIUS,
        max_earth_distance=MOON_ORBIT_RADIUS,
        return_detected=True,
        collision_name=None,
    )

    assert return_type == ReturnType.FREE_RETURN
    assert "Free return confirmed" in diagnostic


def test_ai_preset_returns_to_earth_with_coarse_step() -> None:
    config = with_case(TrajectoryCase.Ai, time_step=180.0)
    result = simulate(config)

    assert result.case == TrajectoryCase.Ai
    assert result.return_type == ReturnType.FREE_RETURN
    assert result.earth_return_time is not None
    assert result.min_moon_distance > 0


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba extra is not installed")
def test_numba_backend_runs_with_public_propagation_api() -> None:
    config = with_case(TrajectoryCase.Ai, duration=600.0)
    initial_state, moon, _ = build_initial_conditions(config)

    result = propagate_trajectory(
        initial_state=initial_state,
        bodies=[EARTH, moon],
        duration=config.duration,
        time_step=60.0,
    )

    assert result.states.shape == (11, 4)
    assert result.collision is None
