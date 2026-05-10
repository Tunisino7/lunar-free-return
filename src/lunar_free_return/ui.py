"""Streamlit interface for exploring lunar free-return configurations."""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lunar_free_return._matplotlib import pyplot as plt
from lunar_free_return.constants import EARTH_RADIUS, MOON_ORBIT_RADIUS, MOON_RADIUS
from lunar_free_return.plotting import (
    FLYBY_COLOR,
    OUTBOUND_COLOR,
    RETURN_COLOR,
    create_figures,
    sampled_indices,
    trajectory_data,
)
from lunar_free_return.simulation import simulate, with_case
from lunar_free_return.types import (
    OrbitalDirection,
    SimulationConfig,
    SimulationResult,
    TrajectoryCase,
)


@dataclass(frozen=True)
class PlaybackSettings:
    """Controls for in-page trajectory playback.

    Parameters
    ----------
    max_frames:
        Maximum number of sampled frames rendered during playback.
    fps:
        Target frames per second used for Streamlit frame updates.
    """

    max_frames: int = 120
    fps: int = 12


def _streamlit_runtime_active() -> bool:
    """Return whether this module is executing inside Streamlit.

    Returns
    -------
    bool
        ``True`` when Streamlit is running this file as an app script;
        ``False`` when the console entry point imports the module normally.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        return False
    return get_script_run_ctx() is not None


def _require_streamlit():
    """Import Streamlit or raise a dependency-specific exit message.

    Returns
    -------
    module
        Imported ``streamlit`` module.

    Raises
    ------
    SystemExit
        If Streamlit is not installed. The message points users to the optional
        ``ui`` extra instead of failing with a raw import traceback.
    """
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        message = (
            "Streamlit is not installed. Install it with: "
            'python -m pip install -e ".[ui]"'
        )
        raise SystemExit(message) from exc
    return st


def _format_direction(direction: OrbitalDirection) -> str:
    """Return the display label for an orbital direction.

    Parameters
    ----------
    direction:
        Orbital direction enum value.

    Returns
    -------
    str
        Human-readable label for the sidebar selector.
    """
    if direction == OrbitalDirection.PROGRADE:
        return "Prograde"
    return "Retrograde"


def _direction_from_label(label: str) -> OrbitalDirection:
    """Convert a direction selector label back to an enum value.

    Parameters
    ----------
    label:
        Sidebar display label.

    Returns
    -------
    OrbitalDirection
        Matching direction enum value.
    """
    if label == "Prograde":
        return OrbitalDirection.PROGRADE
    return OrbitalDirection.RETROGRADE


def _sidebar_controls(st) -> tuple[SimulationConfig, PlaybackSettings, bool]:
    """Build simulation, playback, and action state from sidebar controls.

    Parameters
    ----------
    st:
        Imported Streamlit module.

    Returns
    -------
    tuple[SimulationConfig, PlaybackSettings, bool]
        Selected simulation configuration, playback settings, and whether the
        user requested a new simulation run.
    """
    st.sidebar.header("Configuration")

    mode = st.sidebar.radio(
        "Configuration mode",
        ["Preset", "Manual"],
        horizontal=True,
        help="Use a known Schwaniger case or edit the numerical parameters.",
    )
    case_label = "Trajectory preset" if mode == "Preset" else "Start from preset"
    case = st.sidebar.selectbox(
        case_label,
        [case.value for case in TrajectoryCase],
        index=0,
        help="Schwaniger free-return family. Ai is the Apollo 13-style preset.",
    )
    preset = with_case(case)

    config = preset
    if mode == "Manual":
        st.sidebar.subheader("Manual parameters")

        speed_factor = st.sidebar.slider(
            "Speed factor",
            min_value=0.99000,
            max_value=1.03000,
            value=float(preset.speed_factor),
            step=0.00005,
            format="%.5f",
            help="Multiplier applied to the Hohmann injection speed.",
        )
        moon_phase = st.sidebar.slider(
            "Moon phase adjustment (rad)",
            min_value=-3.20,
            max_value=3.20,
            value=float(preset.moon_phase_adjustment),
            step=0.0002,
            format="%.4f",
            help="Additional lunar phase offset at t = 0.",
        )
        duration_days = st.sidebar.slider(
            "Duration (days)",
            min_value=4.0,
            max_value=25.0,
            value=float(preset.duration / 86400.0),
            step=0.5,
            help="Maximum propagation duration.",
        )
        time_step = st.sidebar.select_slider(
            "Time step (s)",
            options=[30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0],
            value=float(preset.time_step),
            help="RK4 integration step. Smaller values improve accuracy.",
        )

        direction_labels = ["Prograde", "Retrograde"]
        direction = st.sidebar.selectbox(
            "Orbital direction",
            direction_labels,
            index=direction_labels.index(_format_direction(preset.orbital_direction)),
            help="Initial direction around Earth from the injection point.",
        )
        return_threshold = st.sidebar.number_input(
            "Return threshold altitude (km)",
            min_value=1_000.0,
            max_value=50_000.0,
            value=float(preset.return_altitude_threshold_km),
            step=500.0,
            help="Altitude above Earth used to detect the return leg after apogee.",
        )

        config = replace(
            preset,
            speed_factor=float(speed_factor),
            moon_phase_adjustment=float(moon_phase),
            duration=float(duration_days) * 86400.0,
            time_step=float(time_step),
            orbital_direction=_direction_from_label(direction),
            return_altitude_threshold_km=float(return_threshold),
        )
    else:
        st.sidebar.caption(
            "Preset mode uses the published case parameters without overrides."
        )

    st.sidebar.header("Playback")
    playback = PlaybackSettings(
        max_frames=st.sidebar.slider(
            "Frames",
            min_value=40,
            max_value=240,
            value=120,
            step=20,
            help="Number of sampled frames rendered during playback.",
        ),
        fps=st.sidebar.slider(
            "FPS",
            min_value=4,
            max_value=24,
            value=12,
            step=1,
            help="Frame rate used while streaming the trajectory in the app.",
        ),
    )
    run_requested = st.sidebar.button("Run simulation", type="primary")
    return config, playback, run_requested


def _animation_limits(
    probe_km: np.ndarray,
    moon_km: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute stable frame limits for the live trajectory view.

    Parameters
    ----------
    probe_km:
        Probe positions in kilometers with shape ``(N, 2)``.
    moon_km:
        Moon positions in kilometers with shape ``(N, 2)``.

    Returns
    -------
    tuple[float, float, float, float]
        Minimum x, maximum x, minimum y, and maximum y axis limits in
        kilometers.
    """
    x_all = np.concatenate([probe_km[:, 0], moon_km[:, 0]])
    y_all = np.concatenate([probe_km[:, 1], moon_km[:, 1]])
    extent = max(float(x_all.max() - x_all.min()), float(y_all.max() - y_all.min()))
    padding = max(20_000.0, 0.08 * extent)
    return (
        float(x_all.min()) - padding,
        float(x_all.max()) + padding,
        float(y_all.min()) - padding,
        float(y_all.max()) + padding,
    )


def _phase_bounds(result: SimulationResult) -> tuple[int, int]:
    """Return the approximate lunar-flyby phase bounds.

    Parameters
    ----------
    result:
        Simulation result whose apogee index anchors the flyby window.

    Returns
    -------
    tuple[int, int]
        Inclusive start and end indices for the highlighted flyby phase.
    """
    count = len(result.history.states)
    margin = max(1, int(0.10 * count))
    flyby_start = max(0, result.apogee_index - margin)
    flyby_end = min(count - 1, result.apogee_index + margin)
    return (
        flyby_start,
        flyby_end,
    )


def _phase_color(result: SimulationResult, frame_index: int) -> str:
    """Return the trajectory color matching the current mission phase.

    Parameters
    ----------
    result:
        Simulation result used to locate phase boundaries.
    frame_index:
        Current sample index.

    Returns
    -------
    str
        Matplotlib color for outbound, flyby, or return phase.
    """
    flyby_start, flyby_end = _phase_bounds(result)
    if frame_index < flyby_start:
        return OUTBOUND_COLOR
    if frame_index <= flyby_end:
        return FLYBY_COLOR
    return RETURN_COLOR


def _plot_phase_segments(
    axis,
    probe_km: np.ndarray,
    result: SimulationResult,
    *,
    end_index: int | None = None,
    alpha: float,
    linewidth: float,
) -> None:
    """Draw trajectory segments with the same colors as analysis plots.

    Parameters
    ----------
    axis:
        Matplotlib axes receiving the trajectory lines.
    probe_km:
        Probe positions in kilometers with shape ``(N, 2)``.
    result:
        Simulation result used to split the path by mission phase.
    end_index:
        Optional last sample index to draw. ``None`` draws the full path.
    alpha:
        Line opacity.
    linewidth:
        Line width in points.

    Returns
    -------
    None
    """
    count = len(probe_km)
    limit = count - 1 if end_index is None else min(end_index, count - 1)
    flyby_start, flyby_end = _phase_bounds(result)
    segments = (
        (0, flyby_start, OUTBOUND_COLOR, "Outbound"),
        (flyby_start, flyby_end, FLYBY_COLOR, "Lunar flyby"),
        (flyby_end, count - 1, RETURN_COLOR, "Return"),
    )

    for start, end, color, label in segments:
        segment_end = min(end, limit)
        if segment_end < start:
            continue
        points = probe_km[start : segment_end + 1]
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            label=label if end_index is None else None,
        )


def _live_frame_figure(
    result: SimulationResult,
    probe_km: np.ndarray,
    moon_km: np.ndarray,
    days: np.ndarray,
    frame_index: int,
    limits: tuple[float, float, float, float],
    *,
    tail_samples: int = 120,
) -> plt.Figure:
    """Create one live animation frame from trajectory arrays.

    Parameters
    ----------
    result:
        Simulation result used to color trajectory phases.
    probe_km:
        Probe positions in kilometers with shape ``(N, 2)``.
    moon_km:
        Moon positions in kilometers with shape ``(N, 2)``.
    days:
        Simulation times in days.
    frame_index:
        Index of the sample to show as the current spacecraft position.
    limits:
        Axis limits from ``_animation_limits``.
    tail_samples:
        Number of recent samples to emphasize behind the current position.

    Returns
    -------
    plt.Figure
        Matplotlib figure ready for ``st.pyplot`` rendering.
    """
    start = max(0, frame_index - tail_samples)
    figure, axis = plt.subplots(figsize=(8, 8))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#f8fafc")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.grid(True, alpha=0.18)
    axis.set_xlabel("x (km)")
    axis.set_ylabel("y (km)")
    axis.set_title(f"Live trajectory playback - day {days[frame_index]:.2f}")

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    axis.plot(
        MOON_ORBIT_RADIUS / 1e3 * np.cos(theta),
        MOON_ORBIT_RADIUS / 1e3 * np.sin(theta),
        "--",
        color="#a8b3c2",
        linewidth=0.8,
        label="Moon orbit",
    )
    axis.plot(
        moon_km[:, 0],
        moon_km[:, 1],
        color="#94a3b8",
        linewidth=0.9,
        alpha=0.5,
        label="Moon path",
    )

    _plot_phase_segments(axis, probe_km, result, alpha=0.22, linewidth=1.0)
    _plot_phase_segments(
        axis,
        probe_km,
        result,
        end_index=frame_index,
        alpha=0.95,
        linewidth=2.3,
    )
    if start > 0:
        axis.plot(
            probe_km[start : frame_index + 1, 0],
            probe_km[start : frame_index + 1, 1],
            color=_phase_color(result, frame_index),
            linewidth=3.0,
            alpha=0.95,
            solid_capstyle="round",
        )

    axis.scatter(0.0, 0.0, s=90, color="#4a90d9", label="Earth", zorder=4)
    axis.scatter(
        moon_km[frame_index, 0],
        moon_km[frame_index, 1],
        s=60,
        color="#c8cdd4",
        edgecolors="#475569",
        linewidths=0.8,
        label="Moon",
        zorder=4,
    )
    axis.scatter(
        probe_km[frame_index, 0],
        probe_km[frame_index, 1],
        s=48,
        color=_phase_color(result, frame_index),
        edgecolors="#1f2937",
        linewidths=0.8,
        label="Spacecraft",
        zorder=5,
    )
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            EARTH_RADIUS / 1e3,
            color="#4a90d9",
            alpha=0.18,
            zorder=3,
        )
    )
    axis.add_patch(
        plt.Circle(
            tuple(moon_km[frame_index]),
            MOON_RADIUS / 1e3,
            color="#c8cdd4",
            alpha=0.18,
            zorder=3,
        )
    )
    axis.legend(loc="upper right", fontsize=8)
    return figure


def _summary_metrics(st, result) -> None:
    """Render high-level trajectory metrics.

    Parameters
    ----------
    st:
        Imported Streamlit module.
    result:
        Simulation result returned by ``simulate``.

    Returns
    -------
    None
    """
    flyby_altitude_km = (result.min_moon_distance - MOON_RADIUS) / 1e3
    return_days = (
        result.earth_return_time / 86400.0
        if result.earth_return_time is not None
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Classification", str(result.return_type))
    col2.metric("Flyby altitude", f"{flyby_altitude_km:,.0f} km")
    col3.metric("Apogee", f"{result.max_earth_distance / 1e3:,.0f} km")
    col4.metric(
        "Total time", f"{(return_days or result.total_duration / 86400.0):.2f} d"
    )

    st.caption(result.diagnostic)


def _render_figures(st, result) -> None:
    """Render trajectory, distance, and energy figures in tabs.

    Parameters
    ----------
    st:
        Imported Streamlit module.
    result:
        Simulation result returned by ``simulate``.

    Returns
    -------
    None
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        figures = create_figures(result, Path(tmp_dir))
        trajectory_tab, distances_tab, energy_tab = st.tabs(
            ["Trajectory", "Distances", "Energy"]
        )
        for tab, figure in zip(
            [trajectory_tab, distances_tab, energy_tab],
            figures,
            strict=True,
        ):
            with tab:
                st.pyplot(figure, clear_figure=True)
                plt.close(figure)


def _render_live_animation(
    st,
    result: SimulationResult,
    playback: PlaybackSettings,
    *,
    play: bool,
) -> None:
    """Render an in-page trajectory playback without writing media files.

    Parameters
    ----------
    st:
        Imported Streamlit module.
    result:
        Simulation result returned by ``simulate``.
    playback:
        Controls for the number of frames and playback speed.
    play:
        Whether to stream all sampled frames or show the first frame only.

    Returns
    -------
    None
    """
    probe, _, moon, _, days, _ = trajectory_data(result.history, result.moon)
    frame_indices = sampled_indices(len(probe), max_frames=playback.max_frames)
    limits = _animation_limits(probe, moon)

    placeholder = st.empty()
    progress = st.progress(0.0)

    if not play:
        figure = _live_frame_figure(
            result,
            probe,
            moon,
            days,
            int(frame_indices[0]),
            limits,
        )
        placeholder.pyplot(figure, clear_figure=True)
        plt.close(figure)
        return

    delay = 1.0 / float(playback.fps)
    for position, frame_index in enumerate(frame_indices, start=1):
        figure = _live_frame_figure(
            result,
            probe,
            moon,
            days,
            int(frame_index),
            limits,
        )
        placeholder.pyplot(figure, clear_figure=True)
        plt.close(figure)
        progress.progress(position / len(frame_indices))
        time.sleep(delay)


def render_app() -> None:
    """Render the Streamlit application.

    Returns
    -------
    None
    """
    st = _require_streamlit()

    st.set_page_config(page_title="Lunar Free Return", layout="wide")
    st.title("Lunar Free Return")
    st.caption("Explore Schwaniger Earth-Moon free-return trajectories.")

    config, playback, run_requested = _sidebar_controls(st)

    if run_requested or "result" not in st.session_state:
        with st.spinner("Propagating trajectory..."):
            st.session_state.result = simulate(config)
        st.session_state.playback_requested = run_requested

    result = st.session_state.result
    _summary_metrics(st, result)
    animation_tab, analysis_tab = st.tabs(["Live animation", "Analysis plots"])

    with animation_tab:
        play = bool(st.session_state.pop("playback_requested", False))
        _render_live_animation(st, result, playback, play=play)

    with analysis_tab:
        _render_figures(st, result)


def main() -> int:
    """Launch the Streamlit UI from the console script.

    Returns
    -------
    int
        Process exit code from Streamlit.
    """
    _require_streamlit()
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    exit_code = streamlit_cli.main()
    return 0 if exit_code is None else int(exit_code)


if _streamlit_runtime_active():
    render_app()
elif __name__ == "__main__":
    raise SystemExit(main())
