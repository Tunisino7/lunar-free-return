from dataclasses import replace

import numpy as np

from lunar_free_return import (
    OrbitalDirection,
    ReturnType,
    TrajectoryCase,
    build_initial_conditions,
    simulate,
    with_case,
)
from lunar_free_return.constants import EARTH_RADIUS, MOON_ORBIT_RADIUS
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
