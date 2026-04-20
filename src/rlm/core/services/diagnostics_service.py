"""DiagnosticsService — CI/ops oriented runtime checks."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)
from rlm.core.config import load_profile
from rlm.data.backend import DataBackend
from rlm.data.lake.readers import lake_has_bars
from rlm.data.paths import get_data_root, get_processed_data_dir


@dataclass
class CheckResult:
    name: str
    passed: bool | None
    detail: str = ""


@dataclass
class DiagnosticsReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"all_passed": self.all_passed, "checks": [asdict(c) for c in self.checks]}


class DiagnosticsService:
    def run(
        self,
        verbose: bool = False,
        data_root: str | None = None,
        provider: str | None = None,
        backend: str = "auto",
        symbol: str = "SPY",
        mode: str = "plan",
        strict: bool = False,
        backend: str = "auto",
        profile: str | None = None,
        config_path: str | None = None,
        symbol: str | None = None,
    ) -> DiagnosticsReport:
        report = DiagnosticsReport()
        report.checks += self._check_python()
        report.checks += self._check_package_importable()
        report.checks += self._check_core_deps()
        report.checks += self._check_filesystem(data_root)
        report.checks += self._check_config()
        report.checks += self._check_providers(provider)
        report.checks += self._check_ingest_readiness(provider=provider or "yfinance", backend=backend, data_root=data_root)
        report.checks += self._check_trade_readiness(symbol=symbol, backend=backend, mode=mode, data_root=data_root, strict=strict)
        report.checks += self._check_profile(profile, config_path)
        report.checks += self._check_backend(backend, data_root, symbol)
        return report

    def _check_python(self) -> list[CheckResult]:
        v = sys.version_info
        return [CheckResult(name=f"Python {v.major}.{v.minor}.{v.micro}", passed=v >= (3, 10))]

    def _check_package_importable(self) -> list[CheckResult]:
        try:
            import rlm  # noqa: F401
            return [CheckResult(name="package: rlm importable", passed=True)]
        except ImportError as exc:
            return [CheckResult(name="package: rlm importable", passed=False, detail=str(exc))]

    def _check_core_deps(self) -> list[CheckResult]:
        required = ["numpy", "pandas", "pyyaml"]
        out = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
                out.append(CheckResult(name=f"dep: {pkg}", passed=True))
            except ImportError as exc:
                out.append(CheckResult(name=f"dep: {pkg}", passed=False, detail=str(exc)))
        return out

    def _check_filesystem(self, data_root: str | None) -> list[CheckResult]:
        root = get_data_root(data_root)
        processed = get_processed_data_dir(data_root)
        artifacts = root / "artifacts"
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
            artifacts.mkdir(parents=True, exist_ok=True)
            writable_artifacts = os.access(artifacts, os.W_OK)
        except OSError:
            writable_artifacts = False
        return [
            CheckResult(name=f"fs: data root = {root}", passed=True),
            CheckResult(name="fs: processed writable", passed=os.access(processed, os.W_OK), detail=str(processed)),
            CheckResult(name="fs: artifacts writable", passed=writable_artifacts, detail=str(artifacts)),
        ]

    def _check_profile(self, profile: str | None, config_path: str | None) -> list[CheckResult]:
        try:
            if config_path:
                load_profile(path=config_path)
                return [CheckResult(name="config: explicit config valid", passed=True)]
            if profile:
                load_profile(name=profile)
                return [CheckResult(name=f"config: profile '{profile}' valid", passed=True)]
            return [CheckResult(name="config: profile/config", passed=True, detail="default runtime")]
        except Exception as exc:
            return [CheckResult(name="config: profile/config", passed=False, detail=str(exc))]

    def _check_backend(self, backend: str, data_root: str | None, symbol: str | None) -> list[CheckResult]:
        try:
            b = DataBackend.coerce(backend)
        except ValueError as exc:
            return [CheckResult(name="backend: value", passed=False, detail=str(exc))]

        checks = [CheckResult(name=f"backend: {b.value}", passed=True)]
        if symbol:
            sym = symbol.upper()
            raw_csv = get_data_root(data_root) / "raw" / f"bars_{sym}.csv"
            has_csv = raw_csv.is_file()
            has_lake = lake_has_bars(sym, data_root=data_root)
            ok = has_lake if b == DataBackend.LAKE else (has_csv if b == DataBackend.CSV else (has_lake or has_csv))
            checks.append(
                CheckResult(
                    name=f"symbol data: {sym}",
                    passed=ok,
                    detail=f"lake={has_lake} csv={has_csv}",
                )
            )
        return checks
