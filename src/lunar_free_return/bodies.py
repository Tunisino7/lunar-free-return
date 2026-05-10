"""Domain objects for two-dimensional Earth-Moon trajectory propagation.

The public model keeps one ``MassiveBody`` definition. When the optional Numba
extra is installed, that class is decorated into a ``jitclass`` at import time;
otherwise it remains a normal Python class with the same constructor and
methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

StateVector = NDArray[np.float64]
StateHistory = NDArray[np.float64]
ScalarSeries = NDArray[np.float64]

try:  # pragma: no cover - exercised when the optional accel extra is installed.
    from numba import float64, types
    from numba.experimental import jitclass
except ImportError:  # pragma: no cover - default lightweight install path.
    NUMBA_AVAILABLE = False
else:  # pragma: no cover - CI acceleration job exercises this path.
    NUMBA_AVAILABLE = True


class MassiveBody:
    """A fixed or circular-orbit massive body.

    The same class body is used by both runtime modes. With Numba available,
    the class is transformed by ``numba.experimental.jitclass`` using the field
    specification below the class definition. Without Numba, it remains a plain
    Python class. This avoids maintaining a duplicated "compiled" body type.

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

    def __init__(
        self,
        mass: float,
        radius: float,
        x: float,
        y: float,
        angular_rate: float,
        orbital_radius: float,
        phase: float,
        name: str,
    ) -> None:
        self.mass = mass
        self.radius = radius
        self.x = x
        self.y = y
        self.angular_rate = angular_rate
        self.orbital_radius = orbital_radius
        self.phase = phase
        self.name = name

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


if NUMBA_AVAILABLE:
    # Keep this specification adjacent to the class it decorates. The field
    # names intentionally match the Python attributes exactly so the rest of
    # the code can treat both runtime modes identically.
    MassiveBody = jitclass(  # type: ignore[no-redef]
        [
            ("mass", float64),
            ("radius", float64),
            ("x", float64),
            ("y", float64),
            ("angular_rate", float64),
            ("orbital_radius", float64),
            ("phase", float64),
            ("name", types.unicode_type),
        ]
    )(MassiveBody)


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
