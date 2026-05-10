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
    animation_speed:
        Playback speed multiplier applied to the base frame rate.
    """

    max_frames: int = 120
    animation_speed: float = 1.0


BASE_PLAYBACK_FPS = 12.0
LIVE_COMPONENT_HEIGHT = 520


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


def _apply_compact_layout(st) -> None:
    """Apply page-level CSS for the Streamlit UI.

    Parameters
    ----------
    st:
        Imported Streamlit module.

    Returns
    -------
    None
    """
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 0.2rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


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
            "Animation frames",
            min_value=40,
            max_value=240,
            value=120,
            step=20,
            help="Number of sampled frames rendered during playback.",
        ),
        animation_speed=st.sidebar.slider(
            "Animation speed",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.25,
            format="%.2fx",
            help="Playback speed multiplier for the live trajectory animation.",
        ),
    )
    run_requested = st.sidebar.button("Run simulation", type="primary")
    return config, playback, run_requested


def _animation_limits(
    probe_km: np.ndarray,
    moon_km: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute stable frame limits for the live trajectory view.

    The limits must include the full lunar orbit because the SVG draws the
    complete Moon orbit, not only the sampled Moon arc.

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
    moon_orbit_radius_km = MOON_ORBIT_RADIUS / 1e3

    x_all = np.concatenate(
        [
            probe_km[:, 0],
            moon_km[:, 0],
            np.array([-moon_orbit_radius_km, moon_orbit_radius_km]),
        ]
    )
    y_all = np.concatenate(
        [
            probe_km[:, 1],
            moon_km[:, 1],
            np.array([-moon_orbit_radius_km, moon_orbit_radius_km]),
        ]
    )

    x_min = float(x_all.min())
    x_max = float(x_all.max())
    y_min = float(y_all.min())
    y_max = float(y_all.max())

    width = x_max - x_min
    height = y_max - y_min

    # Extra space for bodies, strokes, and the day label.
    padding = max(35_000.0, 0.10 * max(width, height))

    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    return x_min, x_max, y_min, y_max


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


def _summary_values(result: SimulationResult) -> tuple[float, float, float, str]:
    """Return compact values used by the UI summary line.

    Parameters
    ----------
    result:
        Simulation result whose headline metrics should be displayed.

    Returns
    -------
    tuple[float, float, float, str]
        Flyby altitude in kilometers, apogee in kilometers, total shown time in
        days, and return type label.
    """
    flyby_altitude_km = (result.min_moon_distance - MOON_RADIUS) / 1e3
    return_days = (
        result.earth_return_time / 86400.0
        if result.earth_return_time is not None
        else None
    )
    total_days = return_days or result.total_duration / 86400.0
    return (
        flyby_altitude_km,
        result.max_earth_distance / 1e3,
        total_days,
        str(result.return_type),
    )


def _summary_text(result: SimulationResult) -> str:
    """Format the compact trajectory summary.

    Parameters
    ----------
    result:
        Simulation result whose headline metrics should be displayed.

    Returns
    -------
    str
        One-line summary suitable for captions and plot headers.
    """
    flyby_altitude_km, apogee_km, total_days, return_type = _summary_values(result)
    return (
        f"{return_type} | "
        f"Flyby {flyby_altitude_km:,.0f} km | "
        f"Apogee {apogee_km:,.0f} km | "
        f"Total {total_days:.2f} d"
    )


def _svg_path(points: np.ndarray, *, max_points: int = 700) -> str:
    """Convert plot points to a compact SVG path.

    Parameters
    ----------
    points:
        Two-dimensional points in kilometers with shape ``(N, 2)``.
    max_points:
        Maximum number of points kept in the SVG path.

    Returns
    -------
    str
        SVG path data using the same x-axis and an inverted y-axis.
    """
    if len(points) == 0:
        return ""
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    commands = [f"M {points[0, 0]:.1f},{-points[0, 1]:.1f}"]
    commands.extend(f"L {x:.1f},{-y:.1f}" for x, y in points[1:])
    return " ".join(commands)


def _live_frame_svg(
    result: SimulationResult,
    probe_km: np.ndarray,
    moon_km: np.ndarray,
    days: np.ndarray,
    frame_index: int,
    limits: tuple[float, float, float, float],
    *,
    tail_samples: int = 120,
) -> str:
    """Create one responsive SVG animation frame from trajectory arrays.

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
    str
        HTML/SVG markup for ``st.markdown`` rendering.
    """
    start = max(0, frame_index - tail_samples)
    x_min, x_max, y_min, y_max = limits
    width = x_max - x_min
    height = y_max - y_min
    view_box = f"{x_min:.1f} {-y_max:.1f} {width:.1f} {height:.1f}"
    body_radius = 0.008 * max(width, height)
    earth_radius = max(EARTH_RADIUS / 1e3, body_radius)
    moon_radius = max(MOON_RADIUS / 1e3, 0.75 * body_radius)
    probe_radius = 0.45 * body_radius

    flyby_start, flyby_end = _phase_bounds(result)
    segment_specs = (
        (0, flyby_start, OUTBOUND_COLOR),
        (flyby_start, flyby_end, FLYBY_COLOR),
        (flyby_end, len(probe_km) - 1, RETURN_COLOR),
    )
    full_paths = []
    active_paths = []
    for segment_start, segment_end, color in segment_specs:
        points = probe_km[segment_start : segment_end + 1]
        full_paths.append(
            f'<path d="{_svg_path(points)}" stroke="{color}" class="lfr-full-path" />'
        )
        active_end = min(segment_end, frame_index)
        if active_end >= segment_start:
            active_points = probe_km[segment_start : active_end + 1]
            active_paths.append(
                f'<path d="{_svg_path(active_points)}" stroke="{color}" '
                'class="lfr-active-path" />'
            )

    recent_points = probe_km[start : frame_index + 1]
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    orbit = np.column_stack(
        [
            MOON_ORBIT_RADIUS / 1e3 * np.cos(theta),
            MOON_ORBIT_RADIUS / 1e3 * np.sin(theta),
        ]
    )
    current_color = _phase_color(result, frame_index)
    return f"""
    <style>
    html, body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: transparent;
    }}
    .lfr-live-plot {{
        height: {LIVE_COMPONENT_HEIGHT}px;
        width: 100%;
        box-sizing: border-box;
        position: relative;
    }}
    .lfr-live-plot svg {{
        width: 100%;
        height: 100%;
        display: block;
        box-sizing: border-box;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }}
    .lfr-orbit {{
        fill: none;
        stroke: #94a3b8;
        stroke-dasharray: 8 8;
        stroke-width: 1;
        vector-effect: non-scaling-stroke;
    }}
    .lfr-moon-path {{
        fill: none;
        stroke: #94a3b8;
        stroke-width: 1;
        opacity: 0.55;
        vector-effect: non-scaling-stroke;
    }}
    .lfr-full-path {{
        fill: none;
        stroke-width: 1.1;
        opacity: 0.22;
        vector-effect: non-scaling-stroke;
    }}
    .lfr-active-path {{
        fill: none;
        stroke-width: 2.4;
        opacity: 0.95;
        vector-effect: non-scaling-stroke;
    }}
    .lfr-recent-path {{
        fill: none;
        stroke-width: 3.2;
        opacity: 0.95;
        vector-effect: non-scaling-stroke;
    }}
    .lfr-overlay {{
        position: absolute;
        left: 0.55rem;
        top: 0.45rem;
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 0.78rem;
        line-height: 1rem;
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 0.12rem 0.36rem;
    }}
    .lfr-legend {{
        position: absolute;
        right: 0.55rem;
        top: 0.45rem;
        display: flex;
        gap: 0.55rem;
        align-items: center;
        color: #334155;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 0.72rem;
        line-height: 1rem;
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 0.12rem 0.36rem;
    }}
    .lfr-swatch {{
        display: inline-block;
        width: 0.7rem;
        height: 0.14rem;
        margin-right: 0.16rem;
        vertical-align: middle;
    }}
    </style>
    <div class="lfr-live-plot">
      <svg viewBox="{view_box}" preserveAspectRatio="xMidYMid meet"
           role="img" aria-label="Live lunar free-return trajectory">
        <path d="{_svg_path(orbit)}" class="lfr-orbit" />
        <path d="{_svg_path(moon_km)}" class="lfr-moon-path" />
        {"".join(full_paths)}
        {"".join(active_paths)}
        <path d="{_svg_path(recent_points)}" class="lfr-recent-path"
              stroke="{current_color}" />
        <circle cx="0" cy="0" r="{1.8 * earth_radius:.1f}"
                fill="#4a90d9" opacity="0.08" />
        <circle cx="0" cy="0" r="{earth_radius:.1f}"
                fill="#4a90d9" />
        <circle cx="{moon_km[frame_index, 0]:.1f}"
                cy="{-moon_km[frame_index, 1]:.1f}"
                r="{1.8 * moon_radius:.1f}"
                fill="#c8cdd4" opacity="0.10" />
        <circle cx="{moon_km[frame_index, 0]:.1f}"
                cy="{-moon_km[frame_index, 1]:.1f}"
                r="{moon_radius:.1f}"
                fill="#c8cdd4" stroke="#475569"
                stroke-width="1" vector-effect="non-scaling-stroke" />
        <circle cx="{probe_km[frame_index, 0]:.1f}"
                cy="{-probe_km[frame_index, 1]:.1f}"
                r="{probe_radius:.1f}"
                fill="{current_color}" stroke="#1f2937"
                stroke-width="1" vector-effect="non-scaling-stroke" />
      </svg>
      <div class="lfr-overlay">day {days[frame_index]:.2f}</div>
      <div class="lfr-legend">
        <span>
          <i class="lfr-swatch" style="background:{OUTBOUND_COLOR}"></i>Outbound
        </span>
        <span>
          <i class="lfr-swatch" style="background:{FLYBY_COLOR}"></i>Flyby
        </span>
        <span>
          <i class="lfr-swatch" style="background:{RETURN_COLOR}"></i>Return
        </span>
      </div>
    </div>
    """


def _render_live_svg(st, placeholder, svg: str) -> None:
    """Render the live SVG inside a Streamlit iframe.

    Parameters
    ----------
    st:
        Imported Streamlit module.
    placeholder:
        Streamlit placeholder that receives the frame.
    svg:
        Complete HTML/SVG fragment for one animation frame.

    Returns
    -------
    None
    """
    with placeholder.container():
        st.iframe(svg, height=LIVE_COMPONENT_HEIGHT, width="stretch")


def _summary_bar(st, result) -> None:
    """Render compact high-level trajectory metrics with native Streamlit text.

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
    st.markdown("**Lunar Free Return**")
    st.caption(f"{_summary_text(result)} | {result.diagnostic}")


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
                st.pyplot(figure, clear_figure=True, width="stretch")
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

    if not play:
        svg = _live_frame_svg(
            result,
            probe,
            moon,
            days,
            int(frame_indices[0]),
            limits,
        )
        _render_live_svg(st, placeholder, svg)
        return

    progress = st.progress(0.0)
    delay = 1.0 / (BASE_PLAYBACK_FPS * float(playback.animation_speed))
    for position, frame_index in enumerate(frame_indices, start=1):
        svg = _live_frame_svg(
            result,
            probe,
            moon,
            days,
            int(frame_index),
            limits,
        )
        _render_live_svg(st, placeholder, svg)
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
    _apply_compact_layout(st)

    config, playback, run_requested = _sidebar_controls(st)

    if run_requested or "result" not in st.session_state:
        with st.spinner("Propagating trajectory..."):
            st.session_state.result = simulate(config)
        st.session_state.playback_requested = run_requested

    result = st.session_state.result
    _summary_bar(st, result)
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
