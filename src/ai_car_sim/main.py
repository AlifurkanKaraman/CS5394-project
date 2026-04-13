"""Entry point for the ai-car-sim application.

Validates that required dependencies are available, then bootstraps the
simulation by wiring together the menu, training/replay, and engine layers.
"""


def main() -> None:
    """Start the AI car simulation.

    Checks that all required dependencies are installed before proceeding.
    Raises ImportError with a descriptive message if any dependency is missing.
    """
    try:
        import neat  # noqa: F401
    except ImportError:
        raise ImportError(
            "Required dependency 'neat-python' is not installed. "
            "Run: pip install neat-python"
        )

    try:
        import pygame  # noqa: F401
    except ImportError:
        raise ImportError(
            "Required dependency 'pygame' is not installed. "
            "Run: pip install pygame"
        )


if __name__ == "__main__":
    main()
