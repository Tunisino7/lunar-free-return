"""Streamlit interface for exploring lunar free-return configurations."""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from lunar_free_return._matplotlib import pyplot as plt
from lunar_free_return.constants import EARTH_RADIUS, MOON_ORBIT_RADIUS, MOON_RADIUS
from lunar_free_return.plotting import create_figures, sampled_indices, trajectory_data
from lunar_free_return.simulation import simulate, with_case
from lunar_free_return.types import (
    OrbitalDirection,
    SimulationConfig,
    SimulationResult,
    TrajectoryCase,
)


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


def _sidebar_config(st) -> SimulationConfig:
    """Build a simulation configuration from sidebar controls.

    Parameters
    ----------
    st:
        Imported Streamlit module.

    Returns
    -------
    SimulationConfig
        Selected preset configuration or a manually adjusted configuration
        seeded from one preset.
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

    if mode == "Preset":
        st.sidebar.caption(
            "Preset mode uses the published case parameters without overrides."
        )
        return preset

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

    return replace(
        preset,
        speed_factor=float(speed_factor),
        moon_phase_adjustment=float(moon_phase),
        duration=float(duration_days) * 86400.0,
        time_step=float(time_step),
        orbital_direction=_direction_from_label(direction),
        return_altitude_threshold_km=float(return_threshold),
    )


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


def _live_frame_figure(
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
        probe_km[:, 0],
        probe_km[:, 1],
        color="#94a3b8",
        linewidth=0.9,
        alpha=0.35,
        label="Full trajectory",
    )
    axis.plot(
        probe_km[start : frame_index + 1, 0],
        probe_km[start : frame_index + 1, 1],
        color="#0284c7",
        linewidth=2.2,
        label="Current path",
    )
    axis.scatter(0.0, 0.0, s=90, color="#2563eb", label="Earth", zorder=4)
    axis.scatter(
        moon_km[frame_index, 0],
        moon_km[frame_index, 1],
        s=60,
        color="#64748b",
        label="Moon",
        zorder=4,
    )
    axis.scatter(
        probe_km[frame_index, 0],
        probe_km[frame_index, 1],
        s=42,
        color="#dc2626",
        label="Spacecraft",
        zorder=5,
    )
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            EARTH_RADIUS / 1e3,
            color="#2563eb",
            alpha=0.18,
            zorder=3,
        )
    )
    axis.add_patch(
        plt.Circle(
            tuple(moon_km[frame_index]),
            MOON_RADIUS / 1e3,
            color="#64748b",
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


def _render_live_animation(st, result: SimulationResult) -> None:
    """Render an in-page trajectory playback without writing media files.

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
    probe, _, moon, _, days, _ = trajectory_data(result.history, result.moon)
    max_frames = st.slider(
        "Playback frames",
        min_value=40,
        max_value=240,
        value=120,
        step=20,
        help="Number of rendered frames sampled from the trajectory.",
    )
    fps = st.slider(
        "Playback speed (FPS)",
        min_value=4,
        max_value=24,
        value=12,
        step=1,
        help="Frame rate used while streaming the trajectory in the app.",
    )
    frame_indices = sampled_indices(len(probe), max_frames=max_frames)
    limits = _animation_limits(probe, moon)

    placeholder = st.empty()
    progress = st.progress(0.0)
    play = st.button("Play trajectory", type="primary")

    if not play:
        figure = _live_frame_figure(probe, moon, days, int(frame_indices[0]), limits)
        placeholder.pyplot(figure, clear_figure=True)
        plt.close(figure)
        return

    delay = 1.0 / float(fps)
    for position, frame_index in enumerate(frame_indices, start=1):
        figure = _live_frame_figure(probe, moon, days, int(frame_index), limits)
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

    config = _sidebar_config(st)

    if (
        st.sidebar.button("Run simulation", type="primary")
        or "result" not in st.session_state
    ):
        with st.spinner("Propagating trajectory..."):
            st.session_state.result = simulate(config)

    result = st.session_state.result
    _summary_metrics(st, result)
    animation_tab, analysis_tab = st.tabs(["Live animation", "Analysis plots"])

    with animation_tab:
        _render_live_animation(st, result)

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
