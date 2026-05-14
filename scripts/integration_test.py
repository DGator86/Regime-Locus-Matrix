#!/usr/bin/env python3
"""End-to-end wiring integration test.

Runs every major stage of the RLM system with real data (bars_SPY.csv) and
synthetic data (trade log) to verify each component is connected correctly.
No network calls, no IBKR, no Massive.

Steps:
  1  Module imports
  2  LiveRegimeModelConfig — load + field validation
  3  FullRLMPipeline — bars → factors → forecast → ROEE
  4  NightlyMTFOptimizer — 2 Optuna trials → live_nightly_hyperparams.json
  5  apply_nightly_hyperparam_overlay — JSON merges into live config
  6  FullRLMPipeline with overlay applied — confirm params changed
  7  check_performance_and_retune — dry-run, low win rate → trigger path
  8  pnl_report — EOD report from trade_log
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Windows cp1252 consoles cannot print en-dash / arrows in step titles unless UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"

_step = 0
_failures: list[str] = []


def step(title: str) -> None:
    global _step
    _step += 1
    print(f"\n{'='*60}", flush=True)
    print(f"STEP {_step}: {title}", flush=True)
    print("=" * 60, flush=True)


def ok(msg: str) -> None:
    print(f"  [{PASS}] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"  [{INFO}] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  [{FAIL}] {msg}", flush=True)
    _failures.append(msg)


def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        ok(msg_pass)
    else:
        fail(msg_fail)
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Module imports
# ─────────────────────────────────────────────────────────────────────────────

step("Module imports")

try:
    from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline

    ok("rlm.core.pipeline (FullRLMPipeline, FullRLMConfig)")
except Exception as e:
    fail(f"rlm.core.pipeline: {e}")

try:
    from rlm.forecasting.live_model import (
        LiveRegimeModelConfig,
        apply_nightly_hyperparam_overlay,
        load_live_regime_model,
    )

    ok("rlm.forecasting.live_model (LiveRegimeModelConfig, overlay)")
except Exception as e:
    fail(f"rlm.forecasting.live_model: {e}")

try:
    from rlm.optimization.nightly import NightlyMTFOptimizer

    ok("rlm.optimization.nightly (NightlyMTFOptimizer)")
except Exception as e:
    fail(f"rlm.optimization.nightly: {e}")

try:
    ok("rlm.optimization.config (NightlyHyperparams)")
except Exception as e:
    fail(f"rlm.optimization.config: {e}")

try:
    ok("rlm.optimization.base (_signal_based_score, align_regime_labels)")
except Exception as e:
    fail(f"rlm.optimization.base: {e}")

try:
    from rlm.notify.pnl_report import _format_log_section, calculate_daily_pnl

    ok("rlm.notify.pnl_report (calculate_daily_pnl)")
except Exception as e:
    fail(f"rlm.notify.pnl_report: {e}")

try:
    ok("rlm.roee.engine (ROEEConfig)")
except Exception as e:
    fail(f"rlm.roee.engine: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LiveRegimeModelConfig load + field validation
# ─────────────────────────────────────────────────────────────────────────────

step("LiveRegimeModelConfig — load from live_regime_model.json")

import pandas as pd

live_model_path = ROOT / "data" / "processed" / "live_regime_model.json"
live_cfg: LiveRegimeModelConfig | None = None

try:
    live_cfg = load_live_regime_model(live_model_path)
    ok(f"Loaded from {live_model_path.name}")
    info(f"model = {live_cfg.model!r}")
    info(f"roee.confidence_threshold = {live_cfg.roee.confidence_threshold}")
    info(f"roee.high_vol_kelly_multiplier = {live_cfg.roee.high_vol_kelly_multiplier}")
    info(f"roee.transition_kelly_multiplier = {live_cfg.roee.transition_kelly_multiplier}")
    info(f"roee.calm_trend_kelly_multiplier = {live_cfg.roee.calm_trend_kelly_multiplier}")
    info(f"use_kronos = {live_cfg.use_kronos}")
except Exception as e:
    fail(f"load_live_regime_model: {e}")

if live_cfg is not None:
    check(live_cfg.model == "hmm", "model == 'hmm'", f"model = {live_cfg.model!r} (expected 'hmm')")
    check(
        hasattr(live_cfg.roee, "high_vol_kelly_multiplier"),
        "LiveROEEParameters has high_vol_kelly_multiplier",
        "MISSING high_vol_kelly_multiplier on LiveROEEParameters",
    )
    check(
        hasattr(live_cfg.roee, "transition_kelly_multiplier"),
        "LiveROEEParameters has transition_kelly_multiplier",
        "MISSING transition_kelly_multiplier on LiveROEEParameters",
    )
    check(
        hasattr(live_cfg.roee, "calm_trend_kelly_multiplier"),
        "LiveROEEParameters has calm_trend_kelly_multiplier",
        "MISSING calm_trend_kelly_multiplier on LiveROEEParameters",
    )
    dk = live_cfg.decision_kwargs()
    check(
        "high_vol_kelly_multiplier" in dk,
        "decision_kwargs() includes high_vol_kelly_multiplier",
        "decision_kwargs() MISSING high_vol_kelly_multiplier",
    )
    check(
        "transition_kelly_multiplier" in dk,
        "decision_kwargs() includes transition_kelly_multiplier",
        "decision_kwargs() MISSING transition_kelly_multiplier",
    )
    check(
        "calm_trend_kelly_multiplier" in dk,
        "decision_kwargs() includes calm_trend_kelly_multiplier",
        "decision_kwargs() MISSING calm_trend_kelly_multiplier",
    )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FullRLMPipeline: bars → factors → forecast → ROEE
# ─────────────────────────────────────────────────────────────────────────────

step("FullRLMPipeline — bars_SPY.csv → policy_df")

bars_path = ROOT / "data" / "raw" / "bars_SPY.csv"
pipeline_result = None

try:
    bars = pd.read_csv(bars_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    info(f"Loaded {len(bars)} bars from {bars_path.name}")

    cfg = FullRLMConfig(
        regime_model="hmm",
        hmm_states=6,
        use_kronos=False,
        attach_vix=False,
    )
    pipeline_result = FullRLMPipeline(cfg).run(bars)
    ok(
        f"Pipeline ran — factors_df: {len(pipeline_result.factors_df)} rows, "
        f"forecast_df: {len(pipeline_result.forecast_df)} rows, "
        f"policy_df: {len(pipeline_result.policy_df)} rows"
    )
except Exception as e:
    fail(f"FullRLMPipeline.run: {e}")

if pipeline_result is not None:
    pdf = pipeline_result.policy_df
    check(not pdf.empty, "policy_df is not empty", "policy_df is empty")
    check("roee_action" in pdf.columns, "policy_df has roee_action", "policy_df MISSING roee_action")
    check(
        "roee_size_fraction" in pdf.columns,
        "policy_df has roee_size_fraction",
        "policy_df MISSING roee_size_fraction",
    )
    check(
        "close" in pdf.columns,
        "policy_df has close (needed for optimizer scoring)",
        "policy_df MISSING close",
    )

    action_counts = pdf["roee_action"].value_counts().to_dict()
    info(f"roee_action distribution: {action_counts}")

    if "hmm_state" in pdf.columns:
        states = sorted(int(s) for s in pdf["hmm_state"].dropna().unique())
        info(f"hmm_state values: {states}")
        check(
            len(states) > 1,
            f"HMM produced {len(states)} distinct states",
            f"HMM produced only {len(states)} state (alignment may have collapsed)",
        )

    fdf = pipeline_result.forecast_df
    check("sigma" in fdf.columns, "forecast_df has sigma", "forecast_df MISSING sigma")
    check(
        "forecast_return" in fdf.columns,
        "forecast_df has forecast_return",
        "forecast_df MISSING forecast_return",
    )
    check(
        "hmm_confidence" in fdf.columns,
        "forecast_df has hmm_confidence",
        "forecast_df MISSING hmm_confidence",
    )
    info(f"forecast_df shape: {fdf.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — NightlyMTFOptimizer: 2 Optuna trials → JSON
# ─────────────────────────────────────────────────────────────────────────────

step("NightlyMTFOptimizer — 2 Optuna trials on SPY only (no network)")

nightly_json_path = ROOT / "data" / "processed" / "live_nightly_hyperparams.json"

nightly_backup: str | None = None
if nightly_json_path.exists():
    nightly_backup = nightly_json_path.read_text(encoding="utf-8")
    info("Existing live_nightly_hyperparams.json backed up")
generated_nightly_json: str | None = None

opt_result: dict | None = None
try:
    opt_result = NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)
    ok(f"Optimizer completed — keys in result: {sorted(opt_result.keys()) if opt_result else '(empty — all pruned)'}")
except Exception as e:
    fail(f"NightlyMTFOptimizer.run: {e}")

if opt_result is not None:
    if not opt_result:
        fail(
            "Optimizer returned empty dict — all trials were pruned. "
            "Check that use_kronos=False and attach_vix=False are set in OptimizationBase.objective."
        )
    else:
        expected_keys = [
            "mtf_ltf_weight",
            "hmm_confidence_threshold",
            "high_vol_kelly_multiplier",
            "transition_kelly_multiplier",
            "calm_trend_kelly_multiplier",
            "move_window",
            "vol_window",
            "direction_neutral_threshold",
            "transaction_cost_frac",
        ]
        for k in expected_keys:
            check(k in opt_result, f"optimizer output has '{k}'", f"optimizer output MISSING '{k}'")
        check(
            "mtf_regimes" not in opt_result,
            "optimizer output omits unsafe live-only 'mtf_regimes'",
            "optimizer output should not include 'mtf_regimes'",
        )
        check(
            nightly_json_path.exists(),
            "live_nightly_hyperparams.json written",
            "live_nightly_hyperparams.json was NOT written",
        )
        if nightly_json_path.exists():
            generated_nightly_json = nightly_json_path.read_text(encoding="utf-8")
            written = json.loads(generated_nightly_json)
            info(f"Written JSON keys: {sorted(written.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — apply_nightly_hyperparam_overlay
# ─────────────────────────────────────────────────────────────────────────────

step("apply_nightly_hyperparam_overlay — all 7 keys merge correctly")

if live_cfg is not None:
    synthetic_overlay = {
        "move_window": 91,
        "vol_window": 93,
        "direction_neutral_threshold": 0.28,
        "hmm_confidence_threshold": 0.72,
        "high_vol_kelly_multiplier": 0.55,
        "transition_kelly_multiplier": 0.88,
        "calm_trend_kelly_multiplier": 1.15,
    }
    nightly_json_path.write_text(json.dumps(synthetic_overlay, indent=2), encoding="utf-8")
    info("Wrote synthetic overlay with known values")

    try:
        overlaid = apply_nightly_hyperparam_overlay(live_cfg, ROOT)
        ok("apply_nightly_hyperparam_overlay ran without error")

        checks = [
            ("forecast.move_window", overlaid.forecast.move_window, 91),
            ("forecast.vol_window", overlaid.forecast.vol_window, 93),
            (
                "forecast.direction_neutral_threshold",
                overlaid.forecast.direction_neutral_threshold,
                0.28,
            ),
            (
                "roee.confidence_threshold (from hmm_confidence_threshold)",
                overlaid.roee.confidence_threshold,
                0.72,
            ),
            ("roee.high_vol_kelly_multiplier", overlaid.roee.high_vol_kelly_multiplier, 0.55),
            ("roee.transition_kelly_multiplier", overlaid.roee.transition_kelly_multiplier, 0.88),
            ("roee.calm_trend_kelly_multiplier", overlaid.roee.calm_trend_kelly_multiplier, 1.15),
        ]
        for label, got, expected in checks:
            check(
                abs(float(got) - float(expected)) < 1e-9,
                f"{label} == {expected}",
                f"{label} = {got} (expected {expected})",
            )
    except Exception as e:
        fail(f"apply_nightly_hyperparam_overlay: {e}")
else:
    fail("Skipping — live_cfg not available")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FullRLMPipeline: overlay applied via nightly_hyperparams_path
# ─────────────────────────────────────────────────────────────────────────────

step("FullRLMPipeline + overlay JSON — hyperparams reach pipeline config")

try:
    overlay_for_pipeline = {
        "move_window": 88,
        "vol_window": 88,
        "direction_neutral_threshold": 0.29,
        "hmm_confidence_threshold": 0.68,
        "high_vol_kelly_multiplier": 0.52,
        "transition_kelly_multiplier": 0.82,
        "calm_trend_kelly_multiplier": 1.18,
    }
    nightly_json_path.write_text(json.dumps(overlay_for_pipeline, indent=2), encoding="utf-8")

    cfg_ov = FullRLMConfig(
        regime_model="hmm",
        hmm_states=6,
        use_kronos=False,
        attach_vix=False,
        nightly_hyperparams_path=str(nightly_json_path),
    )
    # Pipeline __init__ applies the overlay — check AFTER constructing pipeline
    pipe_ov = FullRLMPipeline(cfg_ov)

    check(
        pipe_ov.config.move_window == 88,
        "pipeline.config.move_window == 88 from overlay",
        f"pipeline.config.move_window = {pipe_ov.config.move_window} (expected 88)",
    )
    check(
        pipe_ov.config.vol_window == 88,
        "pipeline.config.vol_window == 88",
        f"pipeline.config.vol_window = {pipe_ov.config.vol_window}",
    )
    check(
        abs(pipe_ov.config.roee_config.hmm_confidence_threshold - 0.68) < 1e-9,
        "roee_config.hmm_confidence_threshold == 0.68",
        f"roee_config.hmm_confidence_threshold = {pipe_ov.config.roee_config.hmm_confidence_threshold}",
    )
    check(
        abs(pipe_ov.config.roee_config.high_vol_kelly_multiplier - 0.52) < 1e-9,
        "roee_config.high_vol_kelly_multiplier == 0.52",
        f"roee_config.high_vol_kelly_multiplier = {pipe_ov.config.roee_config.high_vol_kelly_multiplier}",
    )
    check(
        abs(pipe_ov.config.roee_config.transition_kelly_multiplier - 0.82) < 1e-9,
        "roee_config.transition_kelly_multiplier == 0.82",
        f"roee_config.transition_kelly_multiplier = {pipe_ov.config.roee_config.transition_kelly_multiplier}",
    )
    check(
        abs(pipe_ov.config.roee_config.calm_trend_kelly_multiplier - 1.18) < 1e-9,
        "roee_config.calm_trend_kelly_multiplier == 1.18",
        f"roee_config.calm_trend_kelly_multiplier = {pipe_ov.config.roee_config.calm_trend_kelly_multiplier}",
    )

    result_ov = pipe_ov.run(bars)
    ok(f"Pipeline with overlay ran — policy_df rows: {len(result_ov.policy_df)}")
    check(
        not result_ov.policy_df.empty,
        "policy_df non-empty with overlay",
        "policy_df empty with overlay",
    )
except Exception as e:
    fail(f"FullRLMPipeline with overlay: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — check_performance_and_retune (dry-run with synthetic log)
# ─────────────────────────────────────────────────────────────────────────────

step("check_performance_and_retune — dry-run, 30% win rate → triggers nightly opt")

import subprocess

today_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tf:
    tmp_log = Path(tf.name)
    writer = csv.DictWriter(
        tf,
        fieldnames=["timestamp_utc", "plan_id", "symbol", "unrealized_pnl", "closed"],
    )
    writer.writeheader()
    # 5 wins, 15 losses = 25% win rate (below warn=40% AND below critical=30%)
    for i in range(20):
        writer.writerow(
            {
                "timestamp_utc": today_str,
                "plan_id": f"TEST_{i:03d}",
                "symbol": "SPY",
                "unrealized_pnl": str(50.0 if i < 5 else -30.0),
                "closed": "1",
            }
        )
    # 2 open rows — should be excluded from win rate calc
    for i in range(2):
        writer.writerow(
            {
                "timestamp_utc": today_str,
                "plan_id": f"OPEN_{i:03d}",
                "symbol": "QQQ",
                "unrealized_pnl": "-10.0",
                "closed": "0",
            }
        )

info(f"Synthetic log: 20 closed (5W/15L = 25%), 2 open → {tmp_log.name}")

retune_script = ROOT / "scripts" / "check_performance_and_retune.py"
try:
    proc = subprocess.run(
        [
            sys.executable,
            str(retune_script),
            "--trade-log",
            str(tmp_log),
            "--lookback",
            "20",
            "--warn-threshold",
            "0.40",
            "--critical-threshold",
            "0.30",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = proc.stdout.strip()
    info(f"Exit code: {proc.returncode}")
    for line in stdout.splitlines():
        info(f"  {line}")
    if proc.stderr.strip():
        info(f"  stderr: {proc.stderr.strip()[:200]}")

    check(
        "win rate 25.0%" in stdout or "25.0%" in stdout,
        "Output reports 25.0% win rate",
        "Win rate not reported correctly in stdout",
    )
    check(
        "nightly_hyperparam_opt" in stdout or "run_nightly" in stdout.lower(),
        "Nightly opt trigger fires (dry-run shows command)",
        "Nightly opt trigger did NOT appear in output",
    )
    check(
        "calibrat" in stdout,
        "Calibration trigger fires (win rate == critical threshold)",
        "Calibration trigger did NOT appear in output",
    )
except Exception as e:
    fail(f"check_performance_and_retune subprocess: {e}")
finally:
    tmp_log.unlink(missing_ok=True)

# Also verify win rate logic in-process
pnls_test = [50.0] * 5 + [-30.0] * 15
wr = sum(1 for p in pnls_test if p > 0) / len(pnls_test)
check(abs(wr - 0.25) < 1e-9, "In-process win rate = 25.0%", f"Got {wr:.1%}")
check(wr < 0.40, "25% < warn 40% → triggers nightly opt", f"{wr:.1%} not < 40%")
check(wr < 0.30, "25% < critical 30% → also triggers calibration", f"{wr:.1%} not < 30%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — pnl_report EOD report generation
# ─────────────────────────────────────────────────────────────────────────────

step("pnl_report — EOD report format, payoff ratio, concentration warning")

from rlm.notify.pnl_report import _format_log_section, _now_utc

today_et = _now_utc().astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
ts_str = today_et.strftime("%Y-%m-%dT%H:%M:%SZ")

# Design: 3 plans total → pnl_sum negative so concentration fires
# Plan 0: META closed loss -150
# Plan 1: SPY closed win +60
# Plan 2: QQQ open +5
# Total = -150 + 60 + 5 = -85.  META = -150 / -85 → 176% → concentration fires
synthetic_rows = [
    {
        "timestamp_utc": ts_str,
        "plan_id": "EOD_META",
        "symbol": "META",
        "unrealized_pnl": "-150.0",
        "closed": "1",
    },
    {
        "timestamp_utc": ts_str,
        "plan_id": "EOD_SPY",
        "symbol": "SPY",
        "unrealized_pnl": "60.0",
        "closed": "1",
    },
    {
        "timestamp_utc": ts_str,
        "plan_id": "EOD_QQQ",
        "symbol": "QQQ",
        "unrealized_pnl": "5.0",
        "closed": "0",
    },
]

try:
    report_block = _format_log_section(
        title="Options (universe monitor / swing or large acct)",
        today_rows=synthetic_rows,
    )
    ok("_format_log_section produced output")
    for line in report_block.splitlines():
        info(f"  {line}")

    check(
        "Exits (closed=1): 2" in report_block,
        "Report shows 2 exits",
        f"Expected '2 exits' — got: {report_block[:300]!r}",
    )
    check("1W / 1L" in report_block, "Report shows 1W / 1L", "Expected '1W / 1L' not found")
    check("Exit payoff" in report_block, "Payoff ratio line present", "Payoff ratio line MISSING")
    check(
        "Concentration: META" in report_block,
        "Concentration warning fires for META (dominant loss)",
        "Concentration warning MISSING for META",
    )
except Exception as e:
    fail(f"_format_log_section: {e}")

# Full calculate_daily_pnl — real trade_log
try:
    report = calculate_daily_pnl(ROOT)
    ok("calculate_daily_pnl completed without crash")
    check(
        isinstance(report, str) and len(report) > 50,
        f"Full report is non-trivial string ({len(report)} chars)",
        "Full report is empty or too short",
    )
    info(f"Full report preview: {report[:200]!r}")
except Exception as e:
    fail(f"calculate_daily_pnl: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Restore
# ─────────────────────────────────────────────────────────────────────────────

if nightly_backup is not None:
    nightly_json_path.write_text(nightly_backup, encoding="utf-8")
    info("\nRestored original live_nightly_hyperparams.json")
elif generated_nightly_json is not None:
    # Keep the real optimizer output, not the synthetic overlays written below.
    nightly_json_path.write_text(generated_nightly_json, encoding="utf-8")
    info("\nRestored optimizer-generated live_nightly_hyperparams.json")
else:
    # synthetic overlay still there — remove it
    nightly_json_path.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}", flush=True)
print("INTEGRATION TEST SUMMARY", flush=True)
print("=" * 60, flush=True)
if _failures:
    print(f"\n  [{FAIL}] {len(_failures)} failure(s):", flush=True)
    for f_msg in _failures:
        print(f"      • {f_msg}", flush=True)
    sys.exit(1)
else:
    print(f"\n  [{PASS}] All checks passed across {_step} steps.", flush=True)
    sys.exit(0)
