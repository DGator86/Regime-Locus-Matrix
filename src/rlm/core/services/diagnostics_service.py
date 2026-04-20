"""DiagnosticsService — preflight environment, dependency, filesystem, and provider checks.

``rlm doctor`` is the authoritative pre-launch health check.  This service
runs a battery of checks and returns a structured report that the CLI renders
and interprets (non-zero exit on any FAIL).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # True=OK  False=FAIL  None=SKIP (optional, not fatal)
    detail: str = ""


@dataclass
class DiagnosticsReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True when no check has ``passed=False``."""
        return all(c.passed is not False for c in self.checks)


class DiagnosticsService:
    """Run a full pre-launch diagnostics battery."""

    def run(
        self,
        verbose: bool = False,
        data_root: str | None = None,
        provider: str | None = None,
        backend: str = "auto",
        symbol: str = "SPY",
        mode: str = "plan",
        strict: bool = False,
    ) -> DiagnosticsReport:
        report = DiagnosticsReport()
        report.checks += self._check_python()
        report.checks += self._check_package_importable()
        report.checks += self._check_core_deps()
        report.checks += self._check_optional_deps()
        report.checks += self._check_env_vars()
        report.checks += self._check_filesystem(data_root)
        report.checks += self._check_config()
        report.checks += self._check_providers(provider)
        report.checks += self._check_ingest_readiness(provider=provider or "yfinance", backend=backend, data_root=data_root)
        report.checks += self._check_trade_readiness(symbol=symbol, backend=backend, mode=mode, data_root=data_root, strict=strict)
        return report

    # ------------------------------------------------------------------
    # Python version
    # ------------------------------------------------------------------

    def _check_python(self) -> list[CheckResult]:
        v = sys.version_info
        ok = v >= (3, 10)
        return [CheckResult(
            name=f"Python {v.major}.{v.minor}.{v.micro}",
            passed=ok,
            detail="" if ok else "Python 3.10+ required",
        )]

    # ------------------------------------------------------------------
    # Package importability
    # ------------------------------------------------------------------

    def _check_package_importable(self) -> list[CheckResult]:
        try:
            import rlm  # noqa: F401
            return [CheckResult(name="package: rlm importable", passed=True)]
        except ImportError as exc:
            return [CheckResult(
                name="package: rlm importable",
                passed=False,
                detail=f"{exc} — run: pip install -e .",
            )]

    # ------------------------------------------------------------------
    # Core dependencies
    # ------------------------------------------------------------------

    def _check_core_deps(self) -> list[CheckResult]:
        required = [
            "numpy", "pandas", "scipy", "statsmodels",
            "hmmlearn", "optuna", "pydantic",
            "yfinance", "matplotlib",
        ]
        results = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
                results.append(CheckResult(name=f"dep: {pkg}", passed=True))
            except ImportError as exc:
                results.append(CheckResult(name=f"dep: {pkg}", passed=False, detail=str(exc)))
        return results

    # ------------------------------------------------------------------
    # Optional extras
    # ------------------------------------------------------------------

    def _check_optional_deps(self) -> list[CheckResult]:
        optional: dict[str, tuple[str, str]] = {
            "torch":      ("kronos",       "pip install -e '.[kronos]'"),
            "duckdb":     ("microstructure", "pip install -e '.[microstructure]'"),
            "streamlit":  ("ui",           "pip install -e '.[ui]'"),
            "ibapi":      ("ibkr",         "pip install ibapi  # manual install"),
            "boto3":      ("flatfiles",    "pip install -e '.[flatfiles]'"),
            "pyarrow":    ("datalake",     "pip install -e '.[datalake]'"),
        }
        results = []
        for pkg, (extra, install_hint) in optional.items():
            try:
                importlib.import_module(pkg)
                results.append(CheckResult(name=f"opt [{extra}]: {pkg}", passed=True))
            except ImportError:
                results.append(CheckResult(
                    name=f"opt [{extra}]: {pkg}",
                    passed=None,  # SKIP — optional, not fatal
                    detail=f"Not installed. {install_hint}",
                ))
        return results

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------

    def _check_env_vars(self) -> list[CheckResult]:
        results: list[CheckResult] = []

        data_root_env = os.environ.get("RLM_DATA_ROOT", "")
        if data_root_env:
            p = Path(data_root_env)
            exists = p.is_dir()
            results.append(CheckResult(
                name=f"env: RLM_DATA_ROOT={data_root_env}",
                passed=exists,
                detail="" if exists else f"Directory does not exist: {p}",
            ))
        else:
            results.append(CheckResult(
                name="env: RLM_DATA_ROOT",
                passed=None,
                detail="Not set — will default to ./data relative to cwd",
            ))

        log_level = os.environ.get("RLM_LOG_LEVEL", "")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level and log_level.upper() not in valid_levels:
            results.append(CheckResult(
                name=f"env: RLM_LOG_LEVEL={log_level}",
                passed=False,
                detail=f"Invalid level; valid: {', '.join(sorted(valid_levels))}",
            ))
        else:
            results.append(CheckResult(
                name=f"env: RLM_LOG_LEVEL={log_level or 'INFO (default)'}",
                passed=True,
            ))

        return results

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------

    def _check_filesystem(self, data_root: str | None) -> list[CheckResult]:
        results: list[CheckResult] = []
        root = get_data_root(data_root)

        results.append(CheckResult(
            name=f"fs: data root = {root}",
            passed=True,
            detail="Resolved from --data-root / RLM_DATA_ROOT / cwd/data",
        ))

        raw_dir = root / "raw"
        results.append(CheckResult(
            name="fs: data/raw/",
            passed=raw_dir.is_dir(),
            detail=(
                "" if raw_dir.is_dir()
                else f"Missing: {raw_dir}\n         Run: rlm ingest --symbol SPY"
            ),
        ))

        processed_dir = root / "processed"
        if not processed_dir.exists():
            try:
                processed_dir.mkdir(parents=True, exist_ok=True)
                writable = True
            except OSError:
                writable = False
        else:
            writable = os.access(processed_dir, os.W_OK)

        results.append(CheckResult(
            name="fs: data/processed/ (writable)",
            passed=writable,
            detail="" if writable else f"Not writable: {processed_dir}",
        ))

        models_dir = root / "models"
        if not models_dir.exists():
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
                m_writable = True
            except OSError:
                m_writable = False
        else:
            m_writable = os.access(models_dir, os.W_OK)

        results.append(CheckResult(
            name="fs: data/models/ (writable)",
            passed=None if not models_dir.exists() else m_writable,
            detail="" if m_writable else f"Not writable: {models_dir}",
        ))

        return results

    # ------------------------------------------------------------------
    # Config construction
    # ------------------------------------------------------------------

    def _check_config(self) -> list[CheckResult]:
        try:
            from rlm.core.pipeline import FullRLMConfig
            cfg = FullRLMConfig()
            assert cfg.regime_model in ("hmm", "markov", "none")
            return [CheckResult(name="config: FullRLMConfig() constructs OK", passed=True)]
        except Exception as exc:
            return [CheckResult(
                name="config: FullRLMConfig() constructs OK",
                passed=False,
                detail=str(exc),
            )]

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _check_providers(self, selected_provider: str | None = None) -> list[CheckResult]:
        results: list[CheckResult] = []

        if selected_provider in (None, "yfinance"):
            try:
                import yfinance as yf
                info = yf.Ticker("SPY").fast_info
                _ = info.last_price
                results.append(CheckResult(name="provider: yfinance (live)", passed=True))
            except Exception as exc:
                results.append(CheckResult(
                    name="provider: yfinance (live)",
                    passed=False,
                    detail=f"Network error or rate-limited: {exc}",
                ))

        # IBKR — import-only (connection requires live TWS)
        if selected_provider in (None, "ibkr"):
            try:
                import ibapi  # noqa: F401
                results.append(CheckResult(
                    name="provider: ibkr (ibapi importable)",
                    passed=True,
                    detail="Connection requires running TWS or IB Gateway",
                ))
            except ImportError:
                results.append(CheckResult(
                    name="provider: ibkr (ibapi importable)",
                    passed=None,
                    detail="ibapi not installed — pip install ibapi",
                ))

        # Massive / boto3 — import-only
        if selected_provider in (None, "massive"):
            try:
                import boto3  # noqa: F401
                results.append(CheckResult(
                    name="provider: massive (boto3 importable)",
                    passed=True,
                    detail="S3 credentials must be set separately (AWS_* env vars)",
                ))
            except ImportError:
                results.append(CheckResult(
                    name="provider: massive (boto3 importable)",
                    passed=None,
                    detail="boto3 not installed — pip install -e '.[flatfiles]'",
                ))

        return results

    def _check_ingest_readiness(self, *, provider: str, backend: str, data_root: str | None) -> list[CheckResult]:
        checks: list[CheckResult] = []
        checks.append(CheckResult(
            name=f"ingest: provider={provider}",
            passed=provider in {"yfinance", "ibkr", "massive"},
            detail="supported: yfinance, ibkr, massive",
        ))
        checks.append(CheckResult(
            name=f"ingest: backend={backend}",
            passed=backend in {"auto", "csv", "lake"},
            detail="supported: auto, csv, lake",
        ))
        root = get_data_root(data_root)
        checks.append(CheckResult(
            name="ingest: artifacts dir writable",
            passed=os.access(root / "artifacts", os.W_OK) if (root / "artifacts").exists() else True,
            detail=str(root / "artifacts"),
        ))
        return checks

    def _check_trade_readiness(
        self,
        *,
        symbol: str,
        backend: str,
        mode: str,
        data_root: str | None,
        strict: bool,
    ) -> list[CheckResult]:
        from rlm.data.readers import _resolve_bars_path

        checks: list[CheckResult] = []
        bars_path = _resolve_bars_path(symbol, None, data_root, backend=backend)
        checks.append(CheckResult(
            name=f"trade: bars available ({symbol})",
            passed=bars_path.exists(),
            detail=str(bars_path),
        ))
        try:
            import rlm.execution.brokers.ibkr_broker  # noqa: F401
            broker_ok = True
        except Exception as exc:
            broker_ok = False
            broker_detail = str(exc)
        else:
            broker_detail = "IBKR broker adapter import OK"
        checks.append(CheckResult(name="trade: broker adapter import", passed=broker_ok, detail=broker_detail))

        if mode in {"paper", "live"}:
            checks.append(CheckResult(
                name=f"trade: mode={mode} prerequisites",
                passed=True if not strict else broker_ok,
                detail="strict mode requires broker adapter import",
            ))
        checks.append(CheckResult(
            name="trade: option-chain availability",
            passed=None,
            detail="warning-only: some strategies may require option-chain data",
        ))
        return checks
