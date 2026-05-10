"""Configuration and result types for free-return trajectory simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

from lunar_free_return.bodies import MassiveBody, SimulationHistory


class OrbitalDirection(IntEnum):
    """Initial orbital direction of the probe around Earth."""

    PROGRADE = 1
    """Same angular direction as the Moon, counterclockwise in this model."""

    RETROGRADE = -1
    """Opposite angular direction from the Moon."""


class ApproachGeometry(StrEnum):
    """Lunar flyby geometry."""

    CIRCUMLUNAR = "circumlunar"
    CISLUNAR = "cislunar"


class TrajectoryCase(StrEnum):
    """Schwaniger's four canonical Earth-Moon free-return cases."""

    Ai = "Ai"
    Aii = "Aii"
    Bi = "Bi"
    Bii = "Bii"


class ReturnType(StrEnum):
    """Qualitative trajectory classification."""

    FREE_RETURN = "free-return"
    DIRECT_RETURN = "direct-return"
    FLYBY_NO_RETURN = "flyby-no-return"
    LUNAR_IMPACT = "lunar-impact"
    ESCAPED = "escaped"


OUTPUT_BASE = Path("output") / "free_return"


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical parameters for one free-return simulation."""

    case: TrajectoryCase = TrajectoryCase.Ai
    orbital_direction: OrbitalDirection = OrbitalDirection.PROGRADE
    departure_altitude: float = 200e3
    time_step: float = 60.0
    duration: float = 12.0 * 86400.0
    output_dir: Path = field(default_factory=lambda: OUTPUT_BASE / "Ai")
    moon_phase_adjustment: float = 0.3499
    speed_factor: float = 1.0037
    return_altitude_threshold_km: float = 10_000.0


PRESET_AI = SimulationConfig(
    case=TrajectoryCase.Ai,
    orbital_direction=OrbitalDirection.PROGRADE,
    speed_factor=1.0037,
    moon_phase_adjustment=0.3499,
    duration=12.0 * 86400.0,
    output_dir=OUTPUT_BASE / "Ai",
)

PRESET_AII = SimulationConfig(
    case=TrajectoryCase.Aii,
    orbital_direction=OrbitalDirection.RETROGRADE,
    speed_factor=1.0015,
    moon_phase_adjustment=0.5000,
    duration=12.0 * 86400.0,
    output_dir=OUTPUT_BASE / "Aii",
)

PRESET_BI = SimulationConfig(
    case=TrajectoryCase.Bi,
    orbital_direction=OrbitalDirection.PROGRADE,
    speed_factor=1.0007,
    moon_phase_adjustment=-0.6000,
    duration=20.0 * 86400.0,
    output_dir=OUTPUT_BASE / "Bi",
)

PRESET_BII = SimulationConfig(
    case=TrajectoryCase.Bii,
    orbital_direction=OrbitalDirection.RETROGRADE,
    speed_factor=1.00045,
    moon_phase_adjustment=-0.5600,
    duration=22.0 * 86400.0,
    output_dir=OUTPUT_BASE / "Bii",
)

PRESETS: dict[TrajectoryCase, SimulationConfig] = {
    TrajectoryCase.Ai: PRESET_AI,
    TrajectoryCase.Aii: PRESET_AII,
    TrajectoryCase.Bi: PRESET_BI,
    TrajectoryCase.Bii: PRESET_BII,
}


@dataclass(frozen=True)
class SimulationResult:
    """Summary and full state history from a free-return simulation."""

    history: SimulationHistory
    moon: MassiveBody
    apogee_index: int
    min_moon_distance: float
    max_earth_distance: float
    earth_return_distance: float | None
    moon_closest_approach_time: float
    earth_return_time: float | None
    total_duration: float
    injection_speed: float
    return_type: ReturnType
    diagnostic: str
    case: TrajectoryCase = TrajectoryCase.Ai
