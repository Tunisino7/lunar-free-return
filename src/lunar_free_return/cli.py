"""Command line interface for the lunar free-return simulator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lunar_free_return.animation import animate
from lunar_free_return.plotting import create_figures, print_summary
from lunar_free_return.simulation import simulate, with_case
from lunar_free_return.types import TrajectoryCase


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with simulation, plotting, and animation options.
    """
    parser = argparse.ArgumentParser(
        description="Simulate Earth-Moon lunar free-return trajectories."
    )
    parser.add_argument(
        "--case",
        choices=[case.value for case in TrajectoryCase],
        default="Ai",
        help="Schwaniger case to simulate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for generated figures and GIFs.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=None,
        help="Integration time step in seconds.",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=None,
        help="Maximum simulation duration in days.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=None,
        help="Multiplier applied to the Hohmann injection speed.",
    )
    parser.add_argument(
        "--moon-phase",
        type=float,
        default=None,
        help="Moon phase adjustment in radians.",
    )
    parser.add_argument(
        "--figures", action="store_true", help="Generate static PNG figures."
    )
    parser.add_argument("--gif", action="store_true", help="Generate an animated GIF.")
    parser.add_argument("--fps", type=int, default=30, help="GIF frames per second.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv:
        Optional argument list. When ``None``, arguments are read from
        ``sys.argv`` by ``argparse``.

    Returns
    -------
    int
        Process exit code. Returns ``0`` after a successful run.
    """
    args = build_parser().parse_args(argv)

    overrides: dict[str, object] = {}
    if args.time_step is not None:
        overrides["time_step"] = args.time_step
    if args.duration_days is not None:
        overrides["duration"] = args.duration_days * 86400.0
    if args.speed_factor is not None:
        overrides["speed_factor"] = args.speed_factor
    if args.moon_phase is not None:
        overrides["moon_phase_adjustment"] = args.moon_phase

    config = with_case(args.case, **overrides)
    output_dir = args.output or config.output_dir
    config = replace(config, output_dir=output_dir)

    result = simulate(config)
    print_summary(result)

    if args.figures:
        create_figures(result, output_dir)
        print(f"Figures written to {output_dir}")
    if args.gif:
        gif_path = animate(result, output_dir, fps=args.fps)
        print(f"Animation written to {gif_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
