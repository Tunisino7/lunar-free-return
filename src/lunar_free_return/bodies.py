"""Domain objects for two-dimensional Earth-Moon trajectory propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

StateVector = NDArray[np.float64]
StateHistory = NDArray[np.float64]
ScalarSeries = NDArray[np.float64]


@dataclass(frozen=True)
class MassiveBody:
    """A fixed or circular-orbit massive body."""

    mass: float
    radius: float
    x: float
    y: float
    angular_rate: float
    orbital_radius: float
    phase: float
    name: str

    def position_x(self, time: float) -> float:
        """Return the x coordinate at ``time`` in meters."""
        if self.angular_rate == 0.0:
            return self.x
        return self.orbital_radius * math.cos(self.angular_rate * time + self.phase)

    def position_y(self, time: float) -> float:
        """Return the y coordinate at ``time`` in meters."""
        if self.angular_rate == 0.0:
            return self.y
        return self.orbital_radius * math.sin(self.angular_rate * time + self.phase)

    def position(self, time: float) -> NDArray[np.float64]:
        """Return the body position at ``time`` as ``[x, y]`` in meters."""
        return np.array([self.position_x(time), self.position_y(time)], dtype=float)


@dataclass(frozen=True)
class SimulationHistory:
    """Full output of a numerical trajectory propagation."""

    times: ScalarSeries
    states: StateHistory
    collision: MassiveBody | None = None
