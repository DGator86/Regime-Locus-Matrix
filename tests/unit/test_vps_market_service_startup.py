from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write_fake_systemctl(bin_dir: Path, log_path: Path) -> None:
    systemctl = bin_dir / "systemctl"
    systemctl.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> {log_path}
if [[ "${{1:-}}" == "list-unit-files" ]]; then
  unit="${{2:-}}"
  case "${{unit}}" in
    rlm-master-trader.service)
      echo "rlm-master-trader.service enabled"
      ;;
    regime-locus-master.service)
      echo "regime-locus-master.service disabled"
      ;;
  esac
fi
exit 0
""",
        encoding="utf-8",
    )
    systemctl.chmod(0o755)


def _run_startup_script(tmp_path: Path, *, forced_window: str) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "systemctl.log"
    _write_fake_systemctl(fake_bin, log_path)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RLM_ROOT": str(tmp_path),
            "RLM_PYTHON": "/nonexistent/python",
            "RLM_FORCE_MARKET_SERVICE_WINDOW": forced_window,
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "rlm_enable_startup_services.sh")],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "starting always-on services" in result.stdout
    return log_path.read_text(encoding="utf-8").splitlines()


def test_startup_helper_does_not_start_trading_units_when_market_window_closed(
    tmp_path: Path,
) -> None:
    commands = _run_startup_script(tmp_path, forced_window="closed")

    assert "disable rlm-master-trader.service" in commands
    assert "disable regime-locus-master.service" in commands
    assert "disable rlm-challenge-loop.service" in commands
    assert "start rlm-host-watchdog.service" in commands
    assert "start regime-locus-crew.service" in commands
    assert "start rlm-master-trader.service" not in commands
    assert "start regime-locus-master.service" not in commands
    assert "start rlm-challenge-loop.service" not in commands


def test_startup_helper_starts_preferred_trading_units_when_market_window_open(
    tmp_path: Path,
) -> None:
    commands = _run_startup_script(tmp_path, forced_window="open")

    assert "start rlm-challenge-loop.service" in commands
    assert "list-unit-files rlm-master-trader.service --no-legend" in commands
    assert "start rlm-master-trader.service" in commands
    assert "start regime-locus-master.service" not in commands


def test_deploy_default_ensure_excludes_market_hours_trading_units() -> None:
    deploy = (ROOT / "scripts" / "deploy_vps.ps1").read_text(encoding="utf-8")

    assert '$ensureRaw = "rlm-systems-control-telegram,rlm-host-watchdog,regime-locus-crew"' in deploy
    assert '$ensureRaw = "rlm-master-trader,rlm-challenge-loop' not in deploy


def test_systemd_installer_does_not_enable_market_hours_trading_units() -> None:
    installer = (ROOT / "deploy" / "linux" / "install-systemd.sh").read_text(encoding="utf-8")

    assert "systemctl enable regime-locus-master.service" not in installer
    assert "systemctl enable rlm-challenge-loop.service" not in installer
    assert "systemctl enable rlm-master-trader.service" not in installer
    assert "systemctl disable regime-locus-master.service rlm-master-trader.service" in installer
