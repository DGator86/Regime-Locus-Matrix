"""RLM CLI entry point — ``rlm <command>``."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    from rlm.utils.compute_threads import apply_compute_thread_env

    apply_compute_thread_env()

    parser = argparse.ArgumentParser(
        prog="rlm",
        description="Regime Locus Matrix — quantitative trading engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  ingest     Fetch and normalize market data into the data lake\n"
            "  forecast   Run the factor + regime + ROEE forecast pipeline\n"
            "  backtest   Execute a strategy backtest (with optional walk-forward)\n"
            "  trade      Generate and execute live/paper trade plans\n"
            "  challenge  $1K->$25K aggressive options dry-run challenge\n"
            "  doctor     Diagnose the environment, providers, and data lake\n"
            "  status     View consolidated PnL across all systems\n"
            "  dashboard  Launch the Streamlit performance dashboard\n"
            "  morning    Run the Morning Briefing protocol (9:00 - 9:45 ET)\n"
        ),
    )
    parser.add_argument("command", choices=["ingest", "forecast", "backtest", "trade", "challenge", "doctor", "status", "dashboard", "morning"])
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    ns = parser.parse_args()

    if ns.command == "ingest":
        from rlm.cli.ingest import main as _main
    elif ns.command == "forecast":
        from rlm.cli.forecast import main as _main  # type: ignore[assignment]
    elif ns.command == "backtest":
        from rlm.cli.backtest import main as _main  # type: ignore[assignment]
    elif ns.command == "trade":
        from rlm.cli.trade import main as _main  # type: ignore[assignment]
    elif ns.command == "challenge":
        from rlm.cli.challenge import main as _main  # type: ignore[assignment]
    elif ns.command == "doctor":
        from rlm.cli.doctor import main as _main  # type: ignore[assignment]
    elif ns.command == "status":
        from rlm.cli.status import main as _main  # type: ignore[assignment]
    elif ns.command == "dashboard":
        import subprocess
        from rlm.data.paths import get_repo_root
        ui_path = get_repo_root() / "src" / "rlm" / "ui" / "dashboard.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
        print(f"Launching dashboard: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    elif ns.command == "morning":
        import subprocess
        from rlm.data.paths import get_repo_root
        script_path = get_repo_root() / "scripts" / "morning_briefing.py"
        cmd = [sys.executable, str(script_path)]
        print(f"Running morning briefing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            pass
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

    # Re-inject argv so each sub-command sees its own args
    sys.argv = [f"rlm {ns.command}", *ns.args]
    _main()


if __name__ == "__main__":
    main()
