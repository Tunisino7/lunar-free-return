"""GIF animation for lunar free-return simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from lunar_free_return.constants import MOON_RADIUS, EARTH_RADIUS
from lunar_free_return.plotting import (
    FLYBY_COLOR,
    OUTBOUND_COLOR,
    RETURN_COLOR,
    configure_space_axis,
    draw_body_with_halos,
    draw_moon_orbit,
    sampled_indices,
    trajectory_data,
)
from lunar_free_return.types import SimulationResult


def _animation_limits(axis, probe_km: np.ndarray, moon_km: np.ndarray) -> None:
    x_all = np.concatenate([probe_km[:, 0], moon_km[:, 0]])
    y_all = np.concatenate([probe_km[:, 1], moon_km[:, 1]])
    extent = max(float(x_all.max() - x_all.min()), float(y_all.max() - y_all.min()))
    padding = 0.1 * extent
    axis.set_xlim(float(x_all.min()) - padding, float(x_all.max()) + padding)
    axis.set_ylim(float(y_all.min()) - padding, float(y_all.max()) + padding)


def animate(result: SimulationResult, output_dir: Path, *, fps: int = 30) -> Path:
    """Create a GIF animation of a free-return trajectory."""
    probe, speeds, moon, moon_distance, days, closest_index = trajectory_data(result.history, result.moon)
    count = len(probe)
    apogee_index = result.apogee_index
    margin = max(1, int(0.10 * count))
    flyby_start = max(0, apogee_index - margin)
    flyby_end = min(count - 1, apogee_index + margin)
    indices = sampled_indices(count, max_frames=300)

    figure, axis = plt.subplots(figsize=(10, 10))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#f7fafc")
    axis.set_aspect("equal", adjustable="box")
    _animation_limits(axis, probe, moon)

    draw_moon_orbit(axis, color="#94a3b8", linewidth=0.7, label=None)
    draw_body_with_halos(axis, (0.0, 0.0), EARTH_RADIUS / 1e3, "#4a90d9", halo_count=4, label="Earth")
    moon_patches = draw_body_with_halos(axis, (0.0, 0.0), MOON_RADIUS / 1e3, "#c8cdd4", halo_count=3, label="Moon")

    outbound_line, = axis.plot([], [], color=OUTBOUND_COLOR, lw=2.0, alpha=0.8)
    flyby_line, = axis.plot([], [], color=FLYBY_COLOR, lw=2.5, alpha=0.9)
    return_line, = axis.plot([], [], color=RETURN_COLOR, lw=2.0, alpha=0.8)
    moon_line, = axis.plot([], [], color="#5c6bc0", lw=0.8, alpha=0.3)
    probe_dot = axis.scatter([], [], s=45, color="white", edgecolors="gray", linewidths=0.8, zorder=6)
    closest_marker = axis.scatter([], [], s=70, marker="D", color=FLYBY_COLOR, edgecolors="#1f2937", linewidths=0.8, zorder=7)
    hud = axis.text(
        0.02,
        0.97,
        "",
        transform=axis.transAxes,
        color="white",
        fontsize=10,
        fontfamily="monospace",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#1f2937", "edgecolor": "#1f2937", "alpha": 0.92},
    )

    for label, color in [("Outbound", OUTBOUND_COLOR), ("Flyby", FLYBY_COLOR), ("Return", RETURN_COLOR)]:
        axis.plot([], [], color=color, lw=2.5, label=label)
    configure_space_axis(axis, f"Lunar free return - case {result.case}", title_fontsize=14)
    axis.legend(loc="lower right", fontsize=9, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")

    def phase_label(index: int) -> str:
        if index < flyby_start:
            return "Outbound"
        if index <= flyby_end:
            return "Lunar flyby"
        return "Return"

    def phase_color(index: int) -> str:
        if index < flyby_start:
            return OUTBOUND_COLOR
        if index <= flyby_end:
            return FLYBY_COLOR
        return RETURN_COLOR

    def init():
        outbound_line.set_data([], [])
        flyby_line.set_data([], [])
        return_line.set_data([], [])
        moon_line.set_data([], [])
        probe_dot.set_offsets(np.empty((0, 2)))
        closest_marker.set_offsets(np.empty((0, 2)))
        for patch in moon_patches:
            patch.set_center((0.0, 0.0))
        hud.set_text("")
        return (outbound_line, flyby_line, return_line, moon_line, probe_dot, *moon_patches, hud, closest_marker)

    def update(frame_index: int):
        index = int(indices[frame_index])

        outbound_end = min(index + 1, flyby_start + 1)
        outbound_line.set_data(probe[:outbound_end, 0], probe[:outbound_end, 1])

        if index > flyby_start:
            flyby_segment_end = min(index + 1, flyby_end + 1)
            segment = probe[flyby_start:flyby_segment_end]
            flyby_line.set_data(segment[:, 0], segment[:, 1])

        if index > flyby_end:
            segment = probe[flyby_end : index + 1]
            return_line.set_data(segment[:, 0], segment[:, 1])

        probe_dot.set_offsets([[probe[index, 0], probe[index, 1]]])
        probe_dot.set_color(phase_color(index))

        for patch in moon_patches:
            patch.set_center((moon[index, 0], moon[index, 1]))
        moon_line.set_data(moon[: index + 1, 0], moon[: index + 1, 1])

        if index >= closest_index:
            closest_marker.set_offsets([[probe[closest_index, 0], probe[closest_index, 1]]])

        hud.set_text(
            f"Phase: {phase_label(index)}\n"
            f"t = {days[index]:.2f} days\n"
            f"v = {speeds[index]:.2f} km/s\n"
            f"d_Moon = {moon_distance[index]:,.0f} km"
        )
        return (outbound_line, flyby_line, return_line, moon_line, probe_dot, *moon_patches, hud, closest_marker)

    animation = FuncAnimation(
        figure,
        update,
        init_func=init,
        frames=len(indices),
        interval=max(1, 1000 // fps),
        blit=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"free_return_{result.case.value}_animation.gif"
    animation.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path
