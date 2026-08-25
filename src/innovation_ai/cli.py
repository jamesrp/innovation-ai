"""Command-line utilities for the development workspace."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from collections.abc import Sequence

from innovation_ai import __version__


def _doctor() -> int:
    """Print a concise environment report."""
    torch_available = importlib.util.find_spec("torch") is not None
    print(f"innovation-ai {__version__}")
    print(f"python {platform.python_version()}")
    print(f"pytorch {'available' if torch_available else 'not installed (run: make install-ai)'}")
    print("device cpu")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(prog="innovation-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="verify the local development environment")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    raise AssertionError(f"unhandled command: {args.command}")
