"""DiagnosticsService — CI/ops oriented runtime checks."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from rlm.core.config import build_pipeline_config, load_profile
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
        *,
        verbose: bool = False,
        data_root: str | None = None,
        provider: str | None = None,
        backend: str = "auto",
        symbol: str | None = None,
        mode: str = "plan",
        strict: bool = False,
        profile: str | None = None,
        config_path: str | None = None,
    ) -> DiagnosticsReport:
        del verbose
        report = DiagnosticsReport()
        report.checks += self._check_python()
        report.checks += self._check_package_importable()
        report.checks += self._check_core_deps()
        report.checks += self._check_filesystem(data_root)
        report.checks += self._check_profile(profile, config_path)
        report.checks += self._check_providers(provider, strict=strict)
        report.checks += self._check_ingest_readiness(
            provider=provider or "yfinance", backend=backend, data_root=data_root
        )
        report.checks += self._check_trade_readiness(
            symbol=symbol, backend=backend, mode=mode, data_root=data_root, strict=strict
        )
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
        required = ["numpy", "pandas", "yaml"]
        out: list[CheckResult] = []
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
        artifacts.mkdir(parents=True, exist_ok=True)
        return [
            CheckResult(name=f"fs: data root = {root}", passed=True),
            CheckResult(
                name="fs: processed writable",
                passed=os.access(processed, os.W_OK),
                detail=str(processed),
            ),
            CheckResult(
                name="fs: artifacts writable",
                passed=os.access(artifacts, os.W_OK),
                detail=str(artifacts),
            ),
        ]

    def _check_providers(
        self, selected_provider: str | None = None, *, strict: bool = False
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        if selected_provider in (None, "yfinance"):
            try:
                import yfinance as yf

                _ = yf.Ticker("SPY")
                results.append(CheckResult(name="provider: yfinance import", passed=True))
            except Exception as exc:
                results.append(
                    CheckResult(name="provider: yfinance import", passed=False, detail=str(exc))
                )

        if selected_provider in (None, "ibkr"):
            try:
                import ibapi  # noqa: F401

                results.append(CheckResult(name="provider: ibkr (ibapi importable)", passed=True))
            except ImportError:
                ibkr_required = strict and selected_provider == "ibkr"
                results.append(
                    CheckResult(
                        name="provider: ibkr (ibapi importable)",
                        passed=False if ibkr_required else None,
                        detail="ibapi not installed",
                    )
                )
        return results

    def _check_ingest_readiness(
        self, *, provider: str, backend: str, data_root: str | None
    ) -> list[CheckResult]:
        from rlm.data.providers import resolve_provider

        root = get_data_root(data_root)
        provider_ok = provider in {"yfinance", "ibkr"}
        provider_detail = "supported: yfinance, ibkr"
        if provider_ok:
            try:
                _ = resolve_provider(provider)
                provider_detail = "provider adapter resolve OK"
            except Exception as exc:
                provider_ok = False
                provider_detail = str(exc)
        return [
            CheckResult(
                name=f"ingest: provider={provider}", passed=provider_ok, detail=provider_detail
            ),
            CheckResult(
                name=f"ingest: backend={backend}",
                passed=backend in {"auto", "csv", "lake"},
                detail="supported: auto, csv, lake",
            ),
            CheckResult(
                name="ingest: artifacts dir writable",
                passed=(
                    os.access(root / "artifacts", os.W_OK)
                    if (root / "artifacts").exists()
                    else True
                ),
                detail=str(root / "artifacts"),
            ),
        ]

    def _check_trade_readiness(
        self,
        *,
        symbol: str | None,
        backend: str,
        mode: str,
        data_root: str | None,
        strict: bool,
    ) -> list[CheckResult]:
        from rlm.data.readers import _resolve_bars_path

        checks: list[CheckResult] = []
        if symbol:
            bars_path = _resolve_bars_path(symbol, None, data_root, backend=backend)
            checks.append(
                CheckResult(
                    name=f"trade: bars available ({symbol})",
                    passed=bars_path.exists(),
                    detail=str(bars_path),
                )
            )
        try:
            import rlm.execution.brokers.ibkr_broker  # noqa: F401

            broker_ok = True
            broker_detail = "IBKR broker adapter import OK"
        except Exception as exc:
            broker_ok = False
            broker_detail = str(exc)
        checks.append(
            CheckResult(name="trade: broker adapter import", passed=broker_ok, detail=broker_detail)
        )
        try:
            _ = build_pipeline_config(
                symbol=symbol or "SPY",
                profile=None,
                config_path=None,
                overrides={"use_kronos": False, "attach_vix": False},
            )
            checks.append(CheckResult(name="trade: config builder resolves", passed=True))
        except Exception as exc:
            checks.append(
                CheckResult(name="trade: config builder resolves", passed=False, detail=str(exc))
            )

        if mode in {"paper", "live"}:
            try:
                import ibapi  # noqa: F401

                deps_ok = True
                deps_detail = "ibapi import OK"
            except Exception as exc:
                deps_ok = False
                deps_detail = str(exc)
            checks.append(
                CheckResult(
                    name=f"trade: mode={mode} broker deps",
                    passed=deps_ok if strict else (deps_ok or None),
                    detail=deps_detail,
                )
            )
            checks.append(
                CheckResult(
                    name=f"trade: mode={mode} prerequisites",
                    passed=True if not strict else (broker_ok and deps_ok),
                    detail="strict mode requires broker adapter and deps",
                )
            )
        return checks

    def _check_profile(self, profile: str | None, config_path: str | None) -> list[CheckResult]:
        try:
            if config_path:
                load_profile(path=config_path)
                return [CheckResult(name="config: explicit config valid", passed=True)]
            if profile:
                load_profile(name=profile)
                return [CheckResult(name=f"config: profile '{profile}' valid", passed=True)]
            return [
                CheckResult(name="config: profile/config", passed=True, detail="default runtime")
            ]
        except Exception as exc:
            return [CheckResult(name="config: profile/config", passed=False, detail=str(exc))]

    def _check_backend(
        self, backend: str, data_root: str | None, symbol: str | None
    ) -> list[CheckResult]:
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
            ok = (
                has_lake
                if b == DataBackend.LAKE
                else (has_csv if b == DataBackend.CSV else (has_lake or has_csv))
            )
            checks.append(
                CheckResult(
                    name=f"symbol data: {sym}", passed=ok, detail=f"lake={has_lake} csv={has_csv}"
                )
            )
        return checks
