"""Earth-Moon lunar free-return trajectory simulator."""

from lunar_free_return.simulation import (
    build_initial_conditions,
    classify_return,
    preset,
    simulate,
    with_case,
)
from lunar_free_return.types import (
    OrbitalDirection,
    ReturnType,
    SimulationConfig,
    SimulationResult,
    TrajectoryCase,
)

__all__ = [
    "OrbitalDirection",
    "ReturnType",
    "SimulationConfig",
    "SimulationResult",
    "TrajectoryCase",
    "build_initial_conditions",
    "classify_return",
    "preset",
    "simulate",
    "with_case",
]
