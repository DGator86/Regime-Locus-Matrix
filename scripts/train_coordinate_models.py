from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rlm.roee.strategy_value_model import STRATEGY_NAMES
from rlm.training.benchmarks import benchmark_coordinate_models, summarize_benchmark_results
from rlm.training.datasets import (
    REQUIRED_COORD_COLUMNS,
    build_regime_training_frame,
    build_strategy_value_training_frame,
)
from rlm.training.feature_ablation import run_temporal_ablation, summarize_ablation
from rlm.training.model_health import evaluate_model_health
from rlm.training.retrain_trigger import should_trigger_retrain
from rlm.training.train_coordinate_models import (
    compute_regime_metrics,
    compute_strategy_metrics,
    load_model_artifacts,
    save_model_artifacts,
    train_regime_model,
    train_strategy_value_model,
)
from rlm.training.validation_matrix import run_validation_matrix, summarize_validation_matrix
from rlm.training.validation_report import save_validation_report


def _load_input_frame(symbols: list[str], data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        path = data_dir / f"features_{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file for {symbol}: {path}")
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _split(df: pd.DataFrame, train_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1)")
    cut = int(len(df) * train_split)
    if cut <= 0 or cut >= len(df):
        raise ValueError("train_split produced empty train or validation partition")
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _load_validation_frames(symbols_csv: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    symbols = [s.strip() for s in symbols_csv.split(",") if s.strip()]
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = data_dir / f"features_{symbol}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        out[symbol] = frame
    return out


def _resolve_health_reference_trained_at(out_dir: Path) -> str:
    """
    Use the most recent persisted artifact timestamp as the age anchor for health checks.
    """
    try:
        regime_artifact, value_artifact = load_model_artifacts(out_dir)
        return max(str(regime_artifact.trained_at), str(value_artifact.trained_at))
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return pd.Timestamp.utcnow().isoformat()


def _selected_strategy_series(
    realized_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if realized_matrix.ndim != 2 or predicted_matrix.ndim != 2:
        raise ValueError("realized_matrix and predicted_matrix must be 2D arrays")
    if realized_matrix.shape != predicted_matrix.shape:
        raise ValueError("realized_matrix and predicted_matrix must have matching shape")
    if realized_matrix.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    selected_idx = predicted_matrix.argmax(axis=1)
    rows = np.arange(realized_matrix.shape[0])
    realized_selected = realized_matrix[rows, selected_idx]
    predicted_selected = predicted_matrix[rows, selected_idx]
    return realized_selected, predicted_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train coordinate regime and strategy value models"
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. SPY,QQQ")
    parser.add_argument("--start", default=None, help="Optional start date filter (inclusive)")
    parser.add_argument("--end", default=None, help="Optional end date filter (inclusive)")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--out-dir", default="artifacts/models")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--label-mode", choices=["bootstrap", "outcome"], default="bootstrap")
    parser.add_argument("--target-mode", choices=["v1", "v2"], default="v2")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--save-benchmark-json", default=None)
    parser.add_argument("--use-path-exits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--sequence-window", type=int, default=None)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--simulator-version", default=None)
    parser.add_argument("--execution-model-version", default=None)
    parser.add_argument("--validation-matrix", action="store_true")
    parser.add_argument("--validation-symbols", default=None)
    parser.add_argument("--validation-windows-json", default=None)
    parser.add_argument("--save-validation-report", default=None)
    parser.add_argument("--enable-persistence-controls", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--evaluate-health", action="store_true")
    parser.add_argument("--auto-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = _load_input_frame(symbols, Path(args.data_dir))

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if args.start:
            df = df.loc[ts >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df.loc[ts <= pd.Timestamp(args.end, tz="UTC")]
        df = df.reset_index(drop=True)

    sequence_window = args.sequence_window if args.temporal else None
    regime_df = build_regime_training_frame(
        df,
        label_mode=args.label_mode,
        horizon=args.horizon,
        sequence_window=sequence_window,
    )
    value_df = build_strategy_value_training_frame(
        df,
        horizon=args.horizon,
        target_mode=args.target_mode,
        use_path_exits=args.use_path_exits,
        sequence_window=sequence_window,
    )

    regime_train, regime_val = _split(regime_df, args.train_split)
    value_train, value_val = _split(value_df, 1.0 - args.val_split)

    regime_model = train_regime_model(regime_train)
    value_model = train_strategy_value_model(value_train)

    regime_metrics = compute_regime_metrics(regime_model, regime_train, regime_val)
    strategy_metrics = compute_strategy_metrics(value_model, value_val)

    benchmark_results = None
    benchmark_summary = None
    if args.benchmark:
        benchmark_results = benchmark_coordinate_models(
            df,
            horizon=args.horizon,
            train_split=args.train_split,
            use_path_exits=args.use_path_exits,
            sequence_window=sequence_window,
            smoothing_alpha=args.smoothing_alpha,
        )
        benchmark_summary = summarize_benchmark_results(benchmark_results)

    validation_results = None
    validation_summary = None
    if args.validation_matrix:
        if not args.validation_symbols or not args.validation_windows_json:
            raise ValueError(
                "validation_matrix requires --validation-symbols and --validation-windows-json"
            )
        frames = _load_validation_frames(args.validation_symbols, Path(args.data_dir))
        windows = json.loads(Path(args.validation_windows_json).read_text(encoding="utf-8"))
        validation_results = run_validation_matrix(
            frames,
            windows=windows,
            horizon=args.horizon,
            sequence_window=args.sequence_window or 5,
            smoothing_alpha=args.smoothing_alpha,
            train_split=args.train_split,
        )
        validation_summary = summarize_validation_matrix(validation_results)
        if args.save_validation_report:
            save_validation_report(validation_results, args.save_validation_report)

    ablation_summary = None
    if args.run_ablation:
        ablation_results = run_temporal_ablation(
            df,
            horizon=args.horizon,
            sequence_window=args.sequence_window or 5,
            smoothing_alpha=args.smoothing_alpha,
            train_split=args.train_split,
        )
        ablation_summary = summarize_ablation(ablation_results)

    training_start = str(df["timestamp"].iloc[0]) if "timestamp" in df.columns and len(df) else None
    training_end = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns and len(df) else None
    model_health_snapshot = None
    health = None
    if args.evaluate_health:
        train_regime_probs = regime_model.predict_proba(regime_train)
        live_regime_probs = regime_model.predict_proba(regime_val)
        x_train = regime_train.loc[:, REQUIRED_COORD_COLUMNS].to_numpy(dtype=float)
        x_live = regime_val.loc[:, REQUIRED_COORD_COLUMNS].to_numpy(dtype=float)
        value_pred_matrix = value_model.predict_expected_values(value_val)
        realized_matrix = value_val.loc[:, STRATEGY_NAMES].to_numpy(dtype=float)
        realized, predicted = _selected_strategy_series(realized_matrix, value_pred_matrix)
        health_trained_at = _resolve_health_reference_trained_at(Path(args.out_dir))
        health = evaluate_model_health(
            trained_at=health_trained_at,
            X_train=x_train,
            X_live=x_live,
            train_regime_probs=train_regime_probs,
            live_regime_probs=live_regime_probs,
            realized_returns=realized,
            predicted_values=predicted,
        )
        model_health_snapshot = {
            "age_hours": float(health.age_hours),
            "feature_drift_score": float(health.feature_drift_score),
            "regime_drift_score": float(health.regime_drift_score),
            "performance_decay": float(health.performance_decay),
            "is_stale": bool(health.is_stale),
        }

    regime_path, value_path = save_model_artifacts(
        regime_model,
        value_model,
        regime_training_rows=len(regime_train),
        value_training_rows=len(value_train),
        source_symbols=symbols,
        out_dir=Path(args.out_dir),
        target_mode=args.target_mode,
        label_mode=args.label_mode,
        horizon=args.horizon,
        training_start=training_start,
        training_end=training_end,
        benchmark_summary=benchmark_summary,
        simulator_version=args.simulator_version,
        execution_model_version=args.execution_model_version,
        train_split=args.train_split,
        validation_rows=len(value_val),
        sequence_window=sequence_window,
        smoothing_alpha=args.smoothing_alpha if args.temporal else None,
        temporal_model=args.temporal,
        validation_matrix_summary=validation_summary,
        feature_ablation_summary=ablation_summary,
        model_health_snapshot=model_health_snapshot,
    )

    if args.save_benchmark_json and benchmark_results is not None:
        out = Path(args.save_benchmark_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")

    print("Regime metrics:")
    print(asdict(regime_metrics))
    print("Strategy metrics:")
    print(asdict(strategy_metrics))
    if benchmark_results is not None:
        print("Benchmark metrics:")
        print(json.dumps(benchmark_results, indent=2))
    if validation_summary is not None:
        print("Validation matrix summary:")
        print(json.dumps(validation_summary, indent=2))
    if ablation_summary is not None:
        print("Temporal ablation summary:")
        print(json.dumps(ablation_summary, indent=2))
    if model_health_snapshot is not None:
        print("Model health:")
        print(json.dumps(model_health_snapshot, indent=2))
        if args.auto_retrain and health is not None and should_trigger_retrain(health):
            print("Triggering retrain...")
            refreshed_regime = train_regime_model(regime_df)
            refreshed_value = train_strategy_value_model(value_df)
            refreshed_regime_path, refreshed_value_path = save_model_artifacts(
                refreshed_regime,
                refreshed_value,
                regime_training_rows=len(regime_df),
                value_training_rows=len(value_df),
                source_symbols=symbols,
                out_dir=Path(args.out_dir),
                target_mode=args.target_mode,
                label_mode=args.label_mode,
                horizon=args.horizon,
                training_start=training_start,
                training_end=training_end,
                benchmark_summary=benchmark_summary,
                simulator_version=args.simulator_version,
                execution_model_version=args.execution_model_version,
                train_split=args.train_split,
                validation_rows=len(value_val),
                sequence_window=sequence_window,
                smoothing_alpha=args.smoothing_alpha if args.temporal else None,
                temporal_model=args.temporal,
                validation_matrix_summary=validation_summary,
                feature_ablation_summary=ablation_summary,
                model_health_snapshot=model_health_snapshot,
            )
            print(f"Auto-retrain completed. Saved regime artifact: {refreshed_regime_path}")
            print(f"Auto-retrain completed. Saved strategy artifact: {refreshed_value_path}")
    print(f"Saved regime artifact: {regime_path}")
    print(f"Saved strategy artifact: {value_path}")


if __name__ == "__main__":
    main()
