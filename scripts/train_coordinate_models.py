from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rlm.training.benchmarks import benchmark_coordinate_models
from rlm.training.datasets import build_regime_training_frame, build_strategy_value_training_frame
from rlm.training.train_coordinate_models import (
    compute_regime_metrics,
    compute_strategy_metrics,
    save_model_artifacts,
    train_regime_model,
    train_strategy_value_model,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train coordinate regime and strategy value models")
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

    regime_df = build_regime_training_frame(df, label_mode=args.label_mode, horizon=args.horizon)
    value_df = build_strategy_value_training_frame(df, horizon=args.horizon, target_mode=args.target_mode)

    regime_train, regime_val = _split(regime_df, args.train_split)
    value_train, value_val = _split(value_df, 1.0 - args.val_split)

    regime_model = train_regime_model(regime_train)
    value_model = train_strategy_value_model(value_train)

    regime_metrics = compute_regime_metrics(regime_model, regime_train, regime_val)
    strategy_metrics = compute_strategy_metrics(value_model, value_val)

    benchmark_results = None
    benchmark_summary = None
    if args.benchmark:
        benchmark_results = benchmark_coordinate_models(df, horizon=args.horizon, train_split=args.train_split)
        baseline = benchmark_results["baseline_b_pr50"]["selected_realized_average"]
        improved = benchmark_results["candidate_d_pr51_full"]["selected_realized_average"]
        benchmark_summary = {"selected_realized_avg_improvement": float(improved - baseline)}

    training_start = str(df["timestamp"].iloc[0]) if "timestamp" in df.columns and len(df) else None
    training_end = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns and len(df) else None
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
    print(f"Saved regime artifact: {regime_path}")
    print(f"Saved strategy artifact: {value_path}")


if __name__ == "__main__":
    main()
