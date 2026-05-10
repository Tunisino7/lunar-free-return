"""Matplotlib backend helpers for non-interactive rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as pyplot  # noqa: E402

__all__ = ["pyplot"]
