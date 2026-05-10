import matplotlib


def test_ui_import_uses_non_interactive_matplotlib_backend() -> None:
    import lunar_free_return.ui  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"
