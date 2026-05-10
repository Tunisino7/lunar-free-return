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
    """A fixed or circular-orbit massive body.

    Parameters
    ----------
    mass:
        Body mass in kilograms.
    radius:
        Physical radius in meters, used for collision detection.
    x, y:
        Fixed Cartesian position in meters. These values are used only when
        ``angular_rate`` is zero.
    angular_rate:
        Circular-orbit angular rate in radians per second. A value of zero
        represents a fixed body.
    orbital_radius:
        Circular-orbit radius in meters around the inertial-frame origin.
    phase:
        Initial orbital phase angle in radians.
    name:
        Human-readable body name used in diagnostics and collision reports.
    """

    mass: float
    radius: float
    x: float
    y: float
    angular_rate: float
    orbital_radius: float
    phase: float
    name: str

    def position_x(self, time: float) -> float:
        """Return the body x coordinate at a simulation time.

        Parameters
        ----------
        time:
            Elapsed simulation time in seconds.

        Returns
        -------
        float
            The x coordinate in meters.
        """
        if self.angular_rate == 0.0:
            return self.x
        return self.orbital_radius * math.cos(self.angular_rate * time + self.phase)

    def position_y(self, time: float) -> float:
        """Return the body y coordinate at a simulation time.

        Parameters
        ----------
        time:
            Elapsed simulation time in seconds.

        Returns
        -------
        float
            The y coordinate in meters.
        """
        if self.angular_rate == 0.0:
            return self.y
        return self.orbital_radius * math.sin(self.angular_rate * time + self.phase)

    def position(self, time: float) -> NDArray[np.float64]:
        """Return the body position vector at a simulation time.

        Parameters
        ----------
        time:
            Elapsed simulation time in seconds.

        Returns
        -------
        NDArray[np.float64]
            Two-element position vector ``[x, y]`` in meters.
        """
        return np.array([self.position_x(time), self.position_y(time)], dtype=float)


@dataclass(frozen=True)
class SimulationHistory:
    """Full output of a numerical trajectory propagation.

    Parameters
    ----------
    times:
        One-dimensional array of sample times in seconds.
    states:
        State history with shape ``(N, 4)``. Each row is
        ``[x, y, vx, vy]`` in meters and meters per second.
    collision:
        First body intersected by the probe, or ``None`` when no collision was
        detected.
    """

    times: ScalarSeries
    states: StateHistory
    collision: MassiveBody | None = None
