"""Installed ContactForge console entry point.

Keep the existing ``python -m contactforge`` CLI as the single source
of truth. This wrapper provides the equivalent ``contactforge`` command.
"""

from contactforge.__main__ import main as _main


def main() -> int:
    """Execute ContactForge's existing command-line interface."""
    return _main()
