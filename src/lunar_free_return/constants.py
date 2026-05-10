"""Physical constants used by the lunar free-return model."""

from __future__ import annotations

import numpy as np

from lunar_free_return.bodies import MassiveBody

G = 6.674e-11

EARTH_MASS = 5.972e24
EARTH_RADIUS = 6.371e6

MOON_MASS = 7.342e22
MOON_RADIUS = 1.737e6
MOON_ORBIT_RADIUS = 3.844e8
MOON_ORBIT_PERIOD = 27.3 * 86400.0
MOON_ANGULAR_RATE = 2.0 * np.pi / MOON_ORBIT_PERIOD

SUN_MASS = 1.989e30

EARTH = MassiveBody(EARTH_MASS, EARTH_RADIUS, 0.0, 0.0, 0.0, 0.0, 0.0, "Earth")
MOON = MassiveBody(
    MOON_MASS,
    MOON_RADIUS,
    0.0,
    0.0,
    MOON_ANGULAR_RATE,
    MOON_ORBIT_RADIUS,
    0.0,
    "Moon",
)
