#!/usr/bin/env python3
"""Regime-locus diagnostics and dashboard generation.

Loads an enriched locus matrix parquet (expected to include ``std_*`` factors,
``S_*`` aggregate scores, and regime columns), then produces:

- Regime transition matrix and dwell-time histograms.
- 2D PCA / t-SNE projections of the full locus (colored by ``hmm_state``).
- P&L attribution by regime and by liquidity-related factors.
- Surface-fit-error correlation plots.
- Interactive Plotly dashboards:
  - regime heatmaps
  - equity curves segmented by confluence score quantiles

Examples::

    python3 scripts/analyze_regime_locus.py --symbol SPY --bar-size 1day
    python3 scripts/analyze_regime_locus.py --symbol QQQ --bar-size 5m \
      --input data/processed/locus_enriched_QQQ_5m.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _clean_bar_size(bar_size: str) -> str:
    return "".join(ch.lower() for ch in bar_size if ch.isalnum())


def _candidate_parquet_paths(symbol: str, bar_size: str) -> list[Path]:
    sym = symbol.upper()
    tag = _clean_bar_size(bar_size)
    return [
        ROOT / f"data/processed/locus_enriched_{sym}_{tag}.parquet",
        ROOT / f"data/processed/locus_matrix_enriched_{sym}_{tag}.parquet",
        ROOT / f"data/processed/locus_matrix_{sym}_{tag}.parquet",
        ROOT / f"data/processed/features_{sym}_{tag}.parquet",
        ROOT / f"data/processed/features_{sym}.parquet",
    ]


def _load_locus(input_path: Path | None, symbol: str, bar_size: str) -> tuple[pd.DataFrame, Path]:
    if input_path is not None:
        p = input_path if input_path.is_absolute() else ROOT / input_path
        if not p.is_file():
            raise FileNotFoundError(f"Input parquet not found: {p}")
        return pd.read_parquet(p), p

    for p in _candidate_parquet_paths(symbol, bar_size):
        if p.is_file():
            return pd.read_parquet(p), p

    tried = "\n".join(f"  - {p}" for p in _candidate_parquet_paths(symbol, bar_size))
    raise FileNotFoundError(
        "Could not find an enriched locus parquet for the requested symbol/bar-size.\n"
        f"Tried:\n{tried}\n"
        "Pass --input explicitly if your path differs."
    )


def _resolve_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.sort_values("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index().reset_index().rename(columns={"index": "timestamp"})
    else:
        out = out.reset_index(drop=True)
        out["timestamp"] = pd.RangeIndex(start=0, stop=len(out), step=1)
    return out


def _pick_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _regime_col(df: pd.DataFrame) -> str:
    col = _pick_first_column(df, ["hmm_state", "hmm_state_label", "regime_key", "regime"])
    if col is None:
        raise ValueError(
            "No regime column found. Expected one of hmm_state/hmm_state_label/regime_key/regime."
        )
    return col


def _pnl_col(df: pd.DataFrame) -> str:
    candidates = [
        "pnl",
        "pnl_net",
        "strategy_pnl",
        "equity_pnl",
        "bar_pnl",
        "trade_pnl",
        "returns_pnl",
    ]
    c = _pick_first_column(df, candidates)
    if c is not None:
        return c

    equity = _pick_first_column(df, ["equity", "portfolio_equity", "account_equity"])
    if equity is not None:
        df["_derived_pnl"] = pd.to_numeric(df[equity], errors="coerce").diff().fillna(0.0)
        return "_derived_pnl"

    if "close" in df.columns:
        df["_derived_pnl"] = pd.to_numeric(df["close"], errors="coerce").pct_change().fillna(0.0)
        return "_derived_pnl"

    raise ValueError("Could not infer P&L column (or derive one from equity/close).")


def _equity_col(df: pd.DataFrame, pnl_col: str) -> str:
    c = _pick_first_column(df, ["equity", "portfolio_equity", "account_equity"])
    if c is not None:
        return c

    seed = 100.0
    df["_derived_equity"] = seed + pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).cumsum()
    return "_derived_equity"


def _confluence_col(df: pd.DataFrame) -> str:
    c = _pick_first_column(
        df,
        ["confluence_score", "signal_confluence", "score_confluence", "confluence"],
    )
    if c is not None:
        return c

    needed = {"S_D", "S_V", "S_L", "S_G"}
    if needed.issubset(df.columns):
        s_d = pd.to_numeric(df["S_D"], errors="coerce").fillna(0.0)
        s_v = pd.to_numeric(df["S_V"], errors="coerce").fillna(0.0)
        s_l = pd.to_numeric(df["S_L"], errors="coerce").fillna(0.0)
        s_g = pd.to_numeric(df["S_G"], errors="coerce").fillna(0.0)
        df["_derived_confluence"] = s_d.abs() + s_g.abs() + s_l - s_v
        return "_derived_confluence"

    df["_derived_confluence"] = 0.0
    return "_derived_confluence"


def _full_locus_numeric(df: pd.DataFrame) -> list[str]:
    base = [c for c in ["S_D", "S_V", "S_L", "S_G"] if c in df.columns]
    std_cols = [c for c in df.columns if c.startswith("std_")]
    cols = base + [c for c in std_cols if c not in base]
    if len(cols) >= 2:
        return cols

    numeric = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"hmm_state"} and not c.endswith("_id")
    ]
    return numeric[: min(len(numeric), 40)]


def _compute_transition(regime: pd.Series) -> pd.DataFrame:
    reg = regime.astype(str)
    prev = reg.shift(1)
    trans = pd.crosstab(prev, reg, normalize="index")
    trans.index.name = "from_regime"
    trans.columns.name = "to_regime"
    return trans.fillna(0.0)


def _compute_dwell(regime: pd.Series) -> pd.DataFrame:
    reg = regime.astype(str)
    run_id = reg.ne(reg.shift(1)).cumsum()
    dwell = (
        pd.DataFrame({"regime": reg, "run_id": run_id})
        .groupby(["regime", "run_id"], as_index=False)
        .size()
        .rename(columns={"size": "dwell_bars"})
    )
    return dwell


def _pca_projection(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=0, how="any")
    if x.empty or x.shape[0] < 3:
        return pd.DataFrame(columns=["pca1", "pca2"])

    values = x.to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    z = (values - mean) / std

    _, _, vt = np.linalg.svd(z, full_matrices=False)
    proj = z @ vt[:2].T
    return pd.DataFrame(proj, index=x.index, columns=["pca1", "pca2"])


def _tsne_projection(df: pd.DataFrame, cols: list[str], random_state: int) -> pd.DataFrame:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return pd.DataFrame(columns=["tsne1", "tsne2"])

    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=0, how="any")
    if x.empty or x.shape[0] < 10:
        return pd.DataFrame(columns=["tsne1", "tsne2"])

    values = x.to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    z = (values - mean) / std

    perplexity = int(max(5, min(40, (len(z) - 1) // 3)))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    out = model.fit_transform(z)
    return pd.DataFrame(out, index=x.index, columns=["tsne1", "tsne2"])


def _liquidity_factor_cols(df: pd.DataFrame) -> list[str]:
    keys = ("liq", "spread", "depth", "open_interest", "oi", "turnover", "slippage", "volume")
    cols: list[str] = []
    for c in df.columns:
        cl = c.lower()
        if c.startswith("std_") and any(k in cl for k in keys):
            cols.append(c)
    return cols


def _safe_write(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--symbol", default="SPY", help="Underlying symbol used for default input/output names."
    )
    parser.add_argument(
        "--bar-size", default="1 day", help="Bar size tag (e.g., '1 day', '5 mins')."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Explicit enriched locus parquet path."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/regime_locus_analysis"),
        help="Directory for reports/plots.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for t-SNE.")
    args = parser.parse_args()

    locus, loaded_path = _load_locus(args.input, args.symbol, args.bar_size)
    df = _resolve_timestamp(locus)

    regime_col = _regime_col(df)
    pnl_col = _pnl_col(df)
    equity_col = _equity_col(df, pnl_col)
    confluence_col = _confluence_col(df)

    out_dir = ROOT / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir = out_dir / args.symbol.upper() / _clean_bar_size(args.bar_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Transition matrix + dwell-time histograms
    trans = _compute_transition(df[regime_col])
    trans.to_csv(out_dir / "regime_transition_matrix.csv")

    dwell = _compute_dwell(df[regime_col])
    dwell.to_csv(out_dir / "regime_dwell_times.csv", index=False)

    plt.figure(figsize=(10, 6))
    for rg, block in dwell.groupby("regime"):
        plt.hist(block["dwell_bars"], bins=20, alpha=0.35, label=str(rg))
    plt.title("Dwell-time histogram by regime")
    plt.xlabel("Bars in run")
    plt.ylabel("Count")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "regime_dwell_histograms.png", dpi=160)
    plt.close()

    # 2) PCA / t-SNE projections
    locus_cols = _full_locus_numeric(df)
    pca = _pca_projection(df, locus_cols)
    tsne = _tsne_projection(df, locus_cols, random_state=args.random_state)

    proj = pd.DataFrame(index=df.index)
    proj = proj.join(pca, how="left").join(tsne, how="left")
    proj["regime"] = df[regime_col].astype(str)
    proj["timestamp"] = df["timestamp"]
    proj.to_csv(out_dir / "locus_projection_points.csv", index=False)

    fig_proj = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "t-SNE"))
    pca_ok = proj[["pca1", "pca2"]].notna().all(axis=1)
    tsne_ok = proj[["tsne1", "tsne2"]].notna().all(axis=1)

    if pca_ok.any():
        for rg in sorted(proj.loc[pca_ok, "regime"].unique()):
            block = proj.loc[pca_ok & (proj["regime"] == rg)]
            fig_proj.add_trace(
                go.Scattergl(
                    x=block["pca1"],
                    y=block["pca2"],
                    mode="markers",
                    marker={"size": 5},
                    name=f"PCA {rg}",
                    legendgroup=str(rg),
                ),
                row=1,
                col=1,
            )

    if tsne_ok.any():
        for rg in sorted(proj.loc[tsne_ok, "regime"].unique()):
            block = proj.loc[tsne_ok & (proj["regime"] == rg)]
            fig_proj.add_trace(
                go.Scattergl(
                    x=block["tsne1"],
                    y=block["tsne2"],
                    mode="markers",
                    marker={"size": 5},
                    name=f"tSNE {rg}",
                    legendgroup=str(rg),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig_proj.update_layout(height=520, width=1200, title_text="Full-locus projection by regime")
    _safe_write(fig_proj, out_dir / "locus_projection_dashboard.html")

    # 3) P&L attribution by regime and liquidity factors
    df["_pnl"] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    by_regime = (
        df.groupby(df[regime_col].astype(str), as_index=False)["_pnl"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(columns={"index": "regime", "sum": "pnl_sum", "mean": "pnl_mean", "count": "bars"})
    )
    by_regime.to_csv(out_dir / "pnl_attribution_by_regime.csv", index=False)

    liq_cols = _liquidity_factor_cols(df)
    liq_rows: list[dict[str, float | str]] = []
    for c in liq_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        y = df["_pnl"]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        corr = float(x[mask].corr(y[mask]))
        slope = float(np.polyfit(x[mask], y[mask], 1)[0])
        liq_rows.append(
            {"factor": c, "corr_with_pnl": corr, "linear_slope": slope, "n": int(mask.sum())}
        )
    liq_attr = (
        pd.DataFrame(liq_rows).sort_values("corr_with_pnl", ascending=False)
        if liq_rows
        else pd.DataFrame()
    )
    liq_attr.to_csv(out_dir / "pnl_attribution_by_liquidity_factors.csv", index=False)

    # 4) Surface-fit-error correlation plots
    sfe_col = _pick_first_column(df, ["surface_fit_error", "std_surface_fit_error"])
    if sfe_col is not None:
        corr_targets = [
            c for c in ["S_D", "S_V", "S_L", "S_G", confluence_col, pnl_col] if c in df.columns
        ]
        corr_rows: list[dict[str, float | str]] = []
        for c in corr_targets:
            x = pd.to_numeric(df[sfe_col], errors="coerce")
            y = pd.to_numeric(df[c], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue
            corr_rows.append(
                {"target": c, "corr": float(x[mask].corr(y[mask])), "n": int(mask.sum())}
            )

            fig = px.scatter(
                df.loc[mask],
                x=sfe_col,
                y=c,
                color=df.loc[mask, regime_col].astype(str),
                trendline="ols",
                title=f"Surface-fit error vs {c}",
            )
            _safe_write(fig, out_dir / f"surface_fit_error_vs_{c}.html")

        pd.DataFrame(corr_rows).to_csv(out_dir / "surface_fit_error_correlations.csv", index=False)

    # 5) Interactive Plotly dashboards (regime heatmaps + confluence-segmented equity)
    heat_figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Transition matrix", "Mean P&L by regime x confluence")
    )
    heat_figure.add_trace(
        go.Heatmap(
            z=trans.values,
            x=[str(c) for c in trans.columns],
            y=[str(i) for i in trans.index],
            colorscale="Viridis",
            colorbar={"title": "P(to | from)"},
        ),
        row=1,
        col=1,
    )

    confluence_numeric = pd.to_numeric(df[confluence_col], errors="coerce")
    try:
        bins = pd.qcut(confluence_numeric, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        bins = pd.Series(["mid"] * len(df), index=df.index)
    df["_confluence_bin"] = bins.astype(str)

    pivot = (
        df.assign(_regime=df[regime_col].astype(str))
        .groupby(["_regime", "_confluence_bin"], as_index=False)["_pnl"]
        .mean()
        .pivot(index="_regime", columns="_confluence_bin", values="_pnl")
        .fillna(0.0)
    )

    heat_figure.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "Mean P&L"},
        ),
        row=1,
        col=2,
    )
    heat_figure.update_layout(height=550, width=1250, title_text="Regime diagnostics heatmaps")
    _safe_write(heat_figure, out_dir / "regime_heatmaps_dashboard.html")

    df["_equity"] = pd.to_numeric(df[equity_col], errors="coerce").ffill().bfill()
    eq_fig = go.Figure()
    for seg in ["low", "mid", "high"]:
        block = df[df["_confluence_bin"] == seg]
        if block.empty:
            continue
        eq_fig.add_trace(
            go.Scatter(
                x=block["timestamp"],
                y=block["_equity"],
                mode="lines",
                name=f"equity ({seg} confluence)",
            )
        )
    eq_fig.update_layout(
        title="Equity curves segmented by confluence",
        xaxis_title="Timestamp",
        yaxis_title="Equity",
        height=520,
        width=1100,
    )
    _safe_write(eq_fig, out_dir / "equity_confluence_dashboard.html")

    summary = {
        "input": str(loaded_path),
        "rows": int(len(df)),
        "regime_column": regime_col,
        "pnl_column": pnl_col,
        "equity_column": equity_col,
        "confluence_column": confluence_col,
        "locus_columns_used": locus_cols,
        "liquidity_factors_used": liq_cols,
        "output_dir": str(out_dir),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded: {loaded_path}")
    print(f"Rows: {len(df):,}")
    print(f"Outputs written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
