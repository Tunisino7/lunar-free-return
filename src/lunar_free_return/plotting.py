"""Static plotting utilities for lunar free-return simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

from lunar_free_return.bodies import MassiveBody, SimulationHistory
from lunar_free_return.constants import (
    G,
    MOON_MASS,
    MOON_ORBIT_RADIUS,
    MOON_RADIUS,
    EARTH_MASS,
    EARTH_RADIUS,
)
from lunar_free_return.types import ReturnType, SimulationResult

OUTBOUND_COLOR = "#4fc3f7"
FLYBY_COLOR = "#ffd54f"
RETURN_COLOR = "#ef5350"


def body_positions(history: SimulationHistory, body: MassiveBody) -> np.ndarray:
    """Compute body positions at every recorded sample time.

    Parameters
    ----------
    history:
        Simulation history providing the time samples.
    body:
        Body whose position should be evaluated.

    Returns
    -------
    np.ndarray
        Array with shape ``(N, 2)`` containing ``[x, y]`` positions in meters.
    """
    return np.array([body.position(t) for t in history.times], dtype=float)


def trajectory_data(
    history: SimulationHistory,
    moon: MassiveBody,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract plot-ready trajectory arrays.

    Parameters
    ----------
    history:
        Propagated probe state history in SI units.
    moon:
        Phased Moon body used to compute lunar positions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        Probe positions in kilometers, probe speeds in kilometers per second,
        Moon positions in kilometers, probe-Moon distances in kilometers, times
        in days, and the closest lunar approach index.
    """
    probe_km = history.states[:, :2] / 1e3
    speeds_km_s = np.linalg.norm(history.states[:, 2:4], axis=1) / 1e3
    moon_km = body_positions(history, moon) / 1e3
    moon_distances_km = np.linalg.norm(probe_km - moon_km, axis=1)
    days = history.times / 86400.0
    closest_index = int(np.argmin(moon_distances_km))
    return probe_km, speeds_km_s, moon_km, moon_distances_km, days, closest_index


def sampled_indices(total: int, max_frames: int = 400) -> np.ndarray:
    """Build a bounded set of frame indices.

    Parameters
    ----------
    total:
        Total number of samples in the source trajectory.
    max_frames:
        Maximum number of indices to return.

    Returns
    -------
    np.ndarray
        Monotonic integer indices that always include the final trajectory
        sample.
    """
    step = max(1, total // min(total, max_frames))
    indices = np.arange(0, total, step)
    if indices[-1] != total - 1:
        indices = np.append(indices, total - 1)
    return indices


def draw_moon_orbit(
    axis: Axes,
    *,
    color: str = "silver",
    linewidth: float = 0.8,
    label: str | None = "Moon orbit",
) -> None:
    """Draw the mean circular lunar orbit in kilometers.

    Parameters
    ----------
    axis:
        Matplotlib axes that receive the orbit line.
    color:
        Line color.
    linewidth:
        Line width in points.
    label:
        Optional legend label. Pass ``None`` to exclude the orbit from the
        legend.

    Returns
    -------
    None
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    axis.plot(
        MOON_ORBIT_RADIUS / 1e3 * np.cos(theta),
        MOON_ORBIT_RADIUS / 1e3 * np.sin(theta),
        "--",
        color=color,
        lw=linewidth,
        label=label,
    )


def draw_body_with_halos(
    axis: Axes,
    center: tuple[float, float],
    radius: float,
    color: str,
    *,
    halo_count: int = 3,
    label: str | None = None,
    zorder: int = 3,
) -> list[Circle]:
    """Draw a body disk with faint visual halos.

    Parameters
    ----------
    axis:
        Matplotlib axes that receive the patches.
    center:
        Body center in kilometers as ``(x, y)``.
    radius:
        Visible disk radius in kilometers.
    color:
        Disk and halo color.
    halo_count:
        Number of translucent halo rings to draw behind the disk.
    label:
        Optional legend label attached to the main disk.
    zorder:
        Drawing order for the main disk.

    Returns
    -------
    list[Circle]
        Halo patches followed by the main disk patch. Animations can update all
        returned patch centers together.
    """
    patches: list[Circle] = []
    for index in range(halo_count, 0, -1):
        halo_radius = radius * (1.0 + index * 0.55)
        alpha = 0.04 + 0.05 * (halo_count - index + 1) / halo_count
        patch = Circle(
            center,
            halo_radius,
            color=color,
            alpha=alpha,
            zorder=zorder - 1,
            linewidth=0,
        )
        axis.add_patch(patch)
        patches.append(patch)
    body = Circle(center, radius, color=color, zorder=zorder, label=label)
    axis.add_patch(body)
    patches.append(body)
    return patches


def configure_space_axis(
    axis: Axes,
    title: str = "",
    *,
    xlabel: str = "x (km)",
    ylabel: str = "y (km)",
    title_fontsize: int = 14,
) -> None:
    """Apply standard formatting for a two-dimensional space plot.

    Parameters
    ----------
    axis:
        Matplotlib axes to configure.
    title:
        Optional axes title.
    xlabel, ylabel:
        Axis labels.
    title_fontsize:
        Title font size in points.

    Returns
    -------
    None
    """
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title, fontsize=title_fontsize, fontweight="bold")
    axis.grid(True, alpha=0.15)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))


def save_figure(figure: Figure, path: Path, *, dpi: int = 200) -> Figure:
    """Save a Matplotlib figure to disk.

    Parameters
    ----------
    figure:
        Figure to save.
    path:
        Destination path. Parent directories are created when needed.
    dpi:
        Output resolution in dots per inch.

    Returns
    -------
    Figure
        The same figure object, allowing callers to keep using or testing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure


def _style_axis(axis: Axes) -> None:
    """Apply the light report style used by all static plots.

    Parameters
    ----------
    axis:
        Matplotlib axes to style.

    Returns
    -------
    None
    """
    axis.set_facecolor("#f8fafc")
    axis.grid(True, alpha=0.32, color="#cbd5e1", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#cbd5e1")
    axis.tick_params(colors="#111827")


def _return_color(return_type: ReturnType) -> str:
    """Map a trajectory classification to a diagnostic color.

    Parameters
    ----------
    return_type:
        Classification from a simulation result.

    Returns
    -------
    str
        Hex color string used for diagnostic box borders.
    """
    if return_type == ReturnType.FREE_RETURN:
        return "#4caf50"
    if return_type == ReturnType.LUNAR_IMPACT:
        return "#e53935"
    if return_type == ReturnType.DIRECT_RETURN:
        return "#ff9800"
    return "#9e9e9e"


def _phase_bounds(result: SimulationResult) -> tuple[int, int]:
    """Compute index bounds for the highlighted lunar-flyby phase.

    Parameters
    ----------
    result:
        Simulation result whose apogee defines the center of the flyby window.

    Returns
    -------
    tuple[int, int]
        Inclusive start and end indices covering a window around apogee.
    """
    count = len(result.history.states)
    margin = max(1, int(0.10 * count))
    start = max(0, result.apogee_index - margin)
    end = min(count - 1, result.apogee_index + margin)
    return start, end


def plot_trajectory(result: SimulationResult, output_path: Path) -> Figure:
    """Plot the full free-return trajectory with mission phases.

    Parameters
    ----------
    result:
        Simulation result to visualize.
    output_path:
        PNG path for the generated trajectory figure.

    Returns
    -------
    Figure
        Saved Matplotlib figure.
    """
    figure, axis = plt.subplots(figsize=(11, 10))
    figure.patch.set_facecolor("white")
    _style_axis(axis)

    probe, _, moon, _, days, closest_index = trajectory_data(result.history, result.moon)
    count = len(probe)
    flyby_start, flyby_end = _phase_bounds(result)

    earth_radius_km = EARTH_RADIUS / 1e3
    moon_radius_km = MOON_RADIUS / 1e3

    draw_moon_orbit(axis, color="#94a3b8", linewidth=0.8, label="Moon orbit")
    axis.plot(moon[:, 0], moon[:, 1], color="#94a3b8", lw=0.9, alpha=0.5, label="Moon path")

    draw_body_with_halos(axis, (0.0, 0.0), earth_radius_km, "#4a90d9", halo_count=4, label="Earth")
    axis.annotate("Earth", (0.0, 0.0), xytext=(12, 12), textcoords="offset points", color="#4a90d9", fontweight="bold")

    moon_x, moon_y = moon[closest_index]
    draw_body_with_halos(axis, (moon_x, moon_y), moon_radius_km, "#c8cdd4", halo_count=3, label="Moon at flyby")
    axis.annotate("Moon", (moon_x, moon_y), xytext=(12, 12), textcoords="offset points", color="#475569", fontweight="bold")

    def segment(start: int, end: int, color: str, label: str) -> None:
        """Draw one colored mission-phase segment on the trajectory plot.

        Parameters
        ----------
        start, end:
            Inclusive trajectory sample indices to draw.
        color:
            Matplotlib color for the segment.
        label:
            Legend label for the segment.

        Returns
        -------
        None
        """
        points = probe[start : end + 1]
        axis.plot(points[:, 0], points[:, 1], color=color, lw=2.2, alpha=0.9, solid_capstyle="round", label=label)

    segment(0, flyby_start, OUTBOUND_COLOR, "Outbound")
    segment(flyby_start, flyby_end, FLYBY_COLOR, "Lunar flyby")
    if flyby_end < count - 1:
        segment(flyby_end, count - 1, RETURN_COLOR, "Return")

    axis.plot(probe[0, 0], probe[0, 1], "o", color="white", mec="#4a90d9", mew=1.5, ms=9, label="LEO departure")
    axis.plot(probe[result.apogee_index, 0], probe[result.apogee_index, 1], "D", color=FLYBY_COLOR, mec="#1f2937", mew=0.8, ms=9, label="Apogee / flyby")

    if result.earth_return_time is not None:
        return_index = int(np.argmin(np.abs(result.history.times - result.earth_return_time)))
        axis.plot(probe[return_index, 0], probe[return_index, 1], "v", color=RETURN_COLOR, mec="black", mew=0.5, ms=11, label="Earth return")

    altitude_km = (result.min_moon_distance - MOON_RADIUS) / 1e3
    outbound_days = result.moon_closest_approach_time / 86400.0
    total_days = (result.earth_return_time or result.total_duration) / 86400.0
    return_days = "none"
    if result.earth_return_time is not None:
        return_days = f"{result.earth_return_time / 86400.0 - outbound_days:.2f} days"

    text = (
        f"Case {result.case} - {result.return_type}\n"
        f"v0 = {result.injection_speed / 1e3:.3f} km/s\n"
        f"lunar flyby altitude = {altitude_km:,.0f} km\n"
        f"apogee = {result.max_earth_distance / 1e3:,.0f} km\n"
        f"outbound time = {outbound_days:.2f} days\n"
        f"return leg = {return_days}\n"
        f"total time = {total_days:.2f} days"
    )
    axis.text(
        0.02,
        0.98,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": _return_color(result.return_type), "linewidth": 1.5},
    )

    configure_space_axis(axis, f"Lunar free-return trajectory - case {result.case}", title_fontsize=13)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.95, facecolor="white", edgecolor="#cbd5e1")
    return save_figure(figure, output_path)


def plot_distances(result: SimulationResult, output_path: Path) -> Figure:
    """Plot Earth and Moon distances through time.

    Parameters
    ----------
    result:
        Simulation result to visualize.
    output_path:
        PNG path for the generated distance figure.

    Returns
    -------
    Figure
        Saved Matplotlib figure.
    """
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    figure.patch.set_facecolor("white")
    for axis in axes:
        _style_axis(axis)

    probe, _, _, moon_distances_km, days, closest_index = trajectory_data(result.history, result.moon)
    earth_distances_km = np.linalg.norm(probe, axis=1)
    flyby_start, flyby_end = _phase_bounds(result)
    count = len(days)

    def plot_phases(axis: Axes, values: np.ndarray) -> None:
        """Plot one time series split into outbound, flyby, and return phases.

        Parameters
        ----------
        axis:
            Matplotlib axes receiving the phase-colored line segments.
        values:
            Time series values aligned with the trajectory sample times.

        Returns
        -------
        None
        """
        for start, end, color in [
            (0, flyby_start, OUTBOUND_COLOR),
            (flyby_start, flyby_end, FLYBY_COLOR),
            (flyby_end, count - 1, RETURN_COLOR),
        ]:
            if end > start:
                axis.plot(days[start : end + 1], values[start : end + 1], color=color, lw=2.0)

    axes[0].axhline(MOON_ORBIT_RADIUS / 1e3, color="#64748b", ls="--", lw=1.0, alpha=0.7, label=f"Moon orbit ({MOON_ORBIT_RADIUS / 1e3:,.0f} km)")
    axes[0].axhline(EARTH_RADIUS / 1e3, color="#4a90d9", ls=":", lw=0.8, alpha=0.6, label=f"Earth surface ({EARTH_RADIUS / 1e3:,.0f} km)")
    plot_phases(axes[0], earth_distances_km)
    axes[0].axvline(days[result.apogee_index], color=FLYBY_COLOR, ls=":", lw=1.2, label=f"Apogee t = {days[result.apogee_index]:.2f} days")
    axes[0].set_ylabel("Distance to Earth (km)")
    axes[0].set_title(f"Distances through time - case {result.case}", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylim(bottom=0)

    axes[1].set_yscale("log")
    plot_phases(axes[1], moon_distances_km)
    axes[1].axhline(MOON_RADIUS / 1e3, color="#c8cdd4", ls=":", lw=0.8, alpha=0.7, label=f"Moon surface ({MOON_RADIUS / 1e3:,.0f} km)")
    axes[1].axvline(days[closest_index], color=FLYBY_COLOR, ls=":", lw=1.2, label=f"Closest flyby t = {days[closest_index]:.2f} days")
    axes[1].plot(days[closest_index], moon_distances_km[closest_index], "D", color=FLYBY_COLOR, ms=8, mec="#1f2937", mew=0.8, label=f"d_min = {moon_distances_km[closest_index]:,.0f} km")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Distance to Moon (km)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.22, which="both", color="#cbd5e1", linewidth=0.7)

    return save_figure(figure, output_path)


def plot_energy(result: SimulationResult, output_path: Path) -> Figure:
    """Plot specific mechanical energy through time.

    Parameters
    ----------
    result:
        Simulation result to evaluate and visualize.
    output_path:
        PNG path for the generated energy figure.

    Returns
    -------
    Figure
        Saved Matplotlib figure.
    """
    history = result.history
    days = history.times / 86400.0
    probe_position = history.states[:, :2]
    probe_velocity = history.states[:, 2:4]
    kinetic = 0.5 * np.sum(probe_velocity**2, axis=1)

    moon_position = body_positions(history, result.moon)
    earth_distance = np.linalg.norm(probe_position, axis=1)
    moon_distance = np.linalg.norm(probe_position - moon_position, axis=1)

    potential = -G * EARTH_MASS / earth_distance - G * MOON_MASS / moon_distance
    total = kinetic + potential
    baseline = total[0]
    relative_drift = (total - baseline) / abs(baseline)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    figure.patch.set_facecolor("white")
    for axis in axes:
        _style_axis(axis)

    axes[0].plot(days, kinetic / 1e6, color=OUTBOUND_COLOR, lw=1.5, label="Kinetic")
    axes[0].plot(days, potential / 1e6, color=RETURN_COLOR, lw=1.5, label="Potential")
    axes[0].plot(days, total / 1e6, color="#2e7d32", lw=2.0, label="Total")
    axes[0].axhline(baseline / 1e6, color="#2e7d32", ls="--", lw=0.8, alpha=0.5)
    flyby_time = result.moon_closest_approach_time / 86400.0
    axes[0].axvline(flyby_time, color=FLYBY_COLOR, ls=":", lw=1.0, alpha=0.7, label="Lunar flyby")
    axes[0].set_ylabel("Specific energy (MJ/kg)")
    axes[0].set_title(f"Specific mechanical energy - case {result.case}", fontsize=13, fontweight="bold")
    axes[0].legend(loc="right", fontsize=9, framealpha=0.95, edgecolor="#cbd5e1")

    axes[1].plot(days, relative_drift * 100, color="#2e7d32", lw=1.2)
    axes[1].axhline(0, color="gray", ls="--", lw=0.7)
    axes[1].axvline(flyby_time, color=FLYBY_COLOR, ls=":", lw=1.0, alpha=0.7)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("dE / |E0| (%)")
    axes[1].text(
        0.98,
        0.92,
        f"max |dE/E0| = {np.max(np.abs(relative_drift)):.2e}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#e8f5e9", "alpha": 0.9},
    )
    return save_figure(figure, output_path)


def create_figures(result: SimulationResult, output_dir: Path) -> list[Figure]:
    """Create all static PNG figures for a simulation result.

    Parameters
    ----------
    result:
        Simulation result to visualize.
    output_dir:
        Directory where figure files are written.

    Returns
    -------
    list[Figure]
        Figures for trajectory, distance history, and energy history.
    """
    case = result.case.value
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_trajectory(result, output_dir / f"free_return_{case}_trajectory.png"),
        plot_distances(result, output_dir / f"free_return_{case}_distances.png"),
        plot_energy(result, output_dir / f"free_return_{case}_energy.png"),
    ]


def print_summary(result: SimulationResult) -> None:
    """Print a compact text summary of a simulation result.

    Parameters
    ----------
    result:
        Simulation result whose key metrics should be printed.

    Returns
    -------
    None
    """
    flyby_days = result.moon_closest_approach_time / 86400.0
    flyby_altitude_km = (result.min_moon_distance - MOON_RADIUS) / 1e3
    return_days = None
    if result.earth_return_time is not None:
        return_days = result.earth_return_time / 86400.0

    print("-" * 58)
    print(f"  Lunar Free-Return Trajectory - case {result.case}")
    print("-" * 58)
    print(f"  Type             : {result.return_type}")
    print(f"  Injection speed  : {result.injection_speed / 1e3:.4f} km/s")
    print(f"  Apogee           : {result.max_earth_distance / 1e3:,.0f} km")
    print(f"  Flyby altitude   : {flyby_altitude_km:,.0f} km above the Moon")
    print(f"  Outbound time    : {flyby_days:.2f} days")
    if return_days is not None:
        print(f"  Return leg       : {return_days - flyby_days:.2f} days")
        print(f"  Total duration   : {return_days:.2f} days")
    else:
        print("  Return leg       : not detected")
        print(f"  Total duration   : {result.total_duration / 86400.0:.2f} days")
    print(f"  Diagnostic       : {result.diagnostic}")
    print("-" * 58)
