"""Command-line utilities for the development workspace."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from innovation_ai import __version__
from innovation_ai.innovation.logs import GameLogError, load_game_log, save_game_log
from innovation_ai.innovation.replay import GameLogRecorder, ReplayError, replay_game_log
from innovation_ai.innovation.state import build_setup_state


def _doctor() -> int:
    """Print a concise environment report."""
    torch_available = importlib.util.find_spec("torch") is not None
    print(f"innovation-ai {__version__}")
    print(f"python {platform.python_version()}")
    print(f"pytorch {'available' if torch_available else 'not installed (run: make install-ai)'}")
    print("device cpu")
    return 0


def _play(seed: int, log_path: Path, max_transitions: int) -> int:
    """Run the current deterministic first-legal-action policy and save its log."""

    recorder = GameLogRecorder(build_setup_state(seed))
    for _ in range(max_transitions):
        decisions = recorder.decisions()
        if not decisions:
            break
        recorder.submit(decisions[0].legal_actions[0])
    game_log = recorder.game_log()
    if game_log.terminal_result is None:
        print(
            f"play stopped without a terminal result after {game_log.transition_count} transitions",
            file=sys.stderr,
        )
        return 2
    save_game_log(game_log, log_path)
    winners = ",".join(player.value for player in game_log.terminal_result.winners) or "draw"
    print(
        f"saved {game_log.transition_count}-transition game to {log_path} "
        f"({game_log.terminal_result.reason.value}: {winners})"
    )
    return 0


def _replay(log_path: Path) -> int:
    """Load and hash-verify one game log."""

    result = replay_game_log(load_game_log(log_path))
    print(
        f"verified {result.transitions_replayed} transitions; "
        f"outcome {result.outcome.value}; hash replay matched"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(prog="innovation-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="verify the local development environment")
    play = subparsers.add_parser("play", help="play a deterministic baseline game and save a log")
    play.add_argument("--seed", type=int, default=0, help="explicit setup seed (default: 0)")
    play.add_argument("--log", type=Path, required=True, help="output game-log path")
    play.add_argument(
        "--max-transitions",
        type=int,
        default=1000,
        help="safety ceiling before refusing to write an incomplete log",
    )
    replay = subparsers.add_parser("replay", help="verify and replay a saved game log")
    replay.add_argument("log", type=Path, help="game-log path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    try:
        if args.command == "play":
            return _play(args.seed, args.log, args.max_transitions)
        if args.command == "replay":
            return _replay(args.log)
    except (GameLogError, ReplayError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")
