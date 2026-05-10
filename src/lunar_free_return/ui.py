"""Streamlit interface for exploring lunar free-return configurations."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from lunar_free_return._matplotlib import pyplot as plt
from lunar_free_return.animation import animate
from lunar_free_return.constants import MOON_RADIUS
from lunar_free_return.plotting import create_figures
from lunar_free_return.simulation import simulate, with_case
from lunar_free_return.types import OrbitalDirection, SimulationConfig, TrajectoryCase


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
        Configuration combining a Schwaniger preset with manual overrides.
    """
    st.sidebar.header("Configuration")

    case = st.sidebar.selectbox(
        "Preset",
        [case.value for case in TrajectoryCase],
        index=0,
        help="Schwaniger free-return family. Ai is the Apollo 13-style preset.",
    )
    preset = with_case(case)

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


def _render_animation(st, result, fps: int) -> None:
    """Generate and render a GIF animation for the current result.

    Parameters
    ----------
    st:
        Imported Streamlit module.
    result:
        Simulation result returned by ``simulate``.
    fps:
        GIF frames per second.

    Returns
    -------
    None
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = animate(result, Path(tmp_dir), fps=fps)
        st.image(path.read_bytes(), caption="Free-return animation")


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
    generate_animation = st.sidebar.checkbox("Generate GIF animation", value=False)
    animation_fps = st.sidebar.slider(
        "GIF FPS",
        min_value=6,
        max_value=30,
        value=12,
        step=1,
        disabled=not generate_animation,
    )

    if (
        st.sidebar.button("Run simulation", type="primary")
        or "result" not in st.session_state
    ):
        with st.spinner("Propagating trajectory..."):
            st.session_state.result = simulate(config)

    result = st.session_state.result
    _summary_metrics(st, result)
    _render_figures(st, result)

    if generate_animation:
        with st.spinner("Rendering GIF..."):
            _render_animation(st, result, animation_fps)


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
