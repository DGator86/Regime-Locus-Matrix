"""Regime cluster analysis: K-Means / Gaussian Mixture + transition matrix + statistics.

Clusters the four-coordinate regime feature space (S_D, S_V, S_L, S_G) to identify
recurring market regimes, compute transition probabilities between them, and measure
per-cluster return characteristics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ClusterMethod = Literal["kmeans", "gmm"]

CLUSTER_COLORS: list[str] = [
    "#00f5ff",  # 0 — cyan
    "#a855f7",  # 1 — purple
    "#00ff9d",  # 2 — green
    "#ff3355",  # 3 — red
    "#ffaa00",  # 4 — orange
    "#88ccff",  # 5 — sky blue
    "#ff88cc",  # 6 — pink
    "#ffff66",  # 7 — yellow
]

DEFAULT_FEATURES: list[str] = ["S_D", "S_V", "S_L", "S_G"]


def cluster_color(cid: int) -> str:
    return CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Core clustering
# ─────────────────────────────────────────────────────────────────────────────


def run_regime_clustering(
    df: pd.DataFrame,
    *,
    n_clusters: int = 4,
    method: ClusterMethod = "kmeans",
    feature_cols: list[str] | None = None,
    lookback: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]] | tuple[None, str]:
    """Cluster the regime feature space using K-Means or Gaussian Mixture.

    Returns ``(clustered_df, feature_cols_used)`` on success, or
    ``(None, error_message)`` on failure.  The returned dataframe is a copy
    of the input (sliced to *lookback* bars if given) with an integer
    ``"cluster"`` column appended.
    """
    try:
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    except ImportError:
        return None, "scikit-learn is not installed — run: pip install scikit-learn"

    cols = feature_cols or DEFAULT_FEATURES
    avail = [
        c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(avail) < 2:
        return None, f"Need ≥ 2 numeric feature columns in the data; found: {avail or 'none'}."

    work = df.copy()
    if lookback and 0 < lookback < len(work):
        work = work.iloc[-lookback:]

    X_raw = work[avail].apply(pd.to_numeric, errors="coerce").dropna()
    min_rows = n_clusters * 3
    if len(X_raw) < min_rows:
        return None, (
            f"Too few clean rows ({len(X_raw)}) for {n_clusters} clusters "
            f"(need ≥ {min_rows}).  Reduce clusters or increase lookback."
        )

    X = StandardScaler().fit_transform(X_raw.values)

    try:
        if method == "kmeans":
            from sklearn.cluster import KMeans  # noqa: PLC0415

            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        else:
            from sklearn.mixture import GaussianMixture  # noqa: PLC0415

            model = GaussianMixture(
                n_components=n_clusters, random_state=random_state, n_init=5
            )
        labels: np.ndarray = model.fit_predict(X)
    except Exception as exc:
        return None, f"Clustering failed: {exc}"

    result = work.loc[X_raw.index].copy()
    result["cluster"] = labels.astype(int)
    return result, avail


# ─────────────────────────────────────────────────────────────────────────────
# Cluster statistics
# ─────────────────────────────────────────────────────────────────────────────


def compute_cluster_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    close_col: str = "close",
) -> pd.DataFrame:
    """Per-cluster centroid, spread, and forward-return statistics.

    Returns a DataFrame indexed by cluster id (0 … n-1) with columns:
    count, pct_of_data, mean_*/std_* for each feature, plus avg_return_pct,
    volatility_pct, and sharpe when close data is present.
    """
    rows: list[dict] = []
    n_total = max(len(df), 1)

    for cid, grp in df.groupby("cluster", sort=True):
        stats: dict = {
            "cluster": int(cid),
            "count": len(grp),
            "pct_of_data": round(len(grp) / n_total * 100, 1),
        }
        for fc in feature_cols:
            vals = pd.to_numeric(grp[fc], errors="coerce").dropna()
            stats[f"mean_{fc}"] = round(float(vals.mean()), 4) if len(vals) else 0.0
            stats[f"std_{fc}"] = round(float(vals.std()), 4) if len(vals) > 1 else 0.0

        if close_col in grp.columns:
            rets = grp[close_col].pct_change().dropna()
            if len(rets):
                mean_r = float(rets.mean())
                std_r = float(rets.std())
                stats["avg_return_pct"] = round(mean_r * 100, 4)
                stats["volatility_pct"] = round(std_r * 100, 4)
                stats["sharpe"] = round(mean_r / std_r, 3) if std_r > 0 else 0.0
        rows.append(stats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("cluster")


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrix
# ─────────────────────────────────────────────────────────────────────────────


def compute_transition_matrix(
    cluster_series: pd.Series,
    n_clusters: int,
) -> pd.DataFrame:
    """Row-normalised cluster-to-cluster transition probability matrix.

    Returns an (n_clusters × n_clusters) DataFrame where entry [i, j] is the
    empirical probability of moving from cluster i to cluster j in one bar.
    """
    labels = cluster_series.dropna().astype(int).values
    mat = np.zeros((n_clusters, n_clusters), dtype=float)
    for i in range(len(labels) - 1):
        f, t = int(labels[i]), int(labels[i + 1])
        if 0 <= f < n_clusters and 0 <= t < n_clusters:
            mat[f, t] += 1.0

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob = mat / row_sums
    idx = [f"C{i}" for i in range(n_clusters)]
    return pd.DataFrame(prob, index=idx, columns=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def regime_label(stats_row: pd.Series) -> str:
    """Short human-readable regime label from a cluster centroid row."""
    parts: list[str] = []
    if "mean_S_D" in stats_row:
        sd = float(stats_row["mean_S_D"])
        parts.append("Bull" if sd > 0.4 else ("Bear" if sd < -0.4 else "Neutral"))
    if "mean_S_V" in stats_row:
        sv = float(stats_row["mean_S_V"])
        parts.append("Hi-Vol" if sv > 0.3 else ("Lo-Vol" if sv < -0.3 else "Mod-Vol"))
    if "mean_S_L" in stats_row:
        sl = float(stats_row["mean_S_L"])
        parts.append("Liquid" if sl > 0.3 else ("Illiquid" if sl < -0.3 else ""))
    return " / ".join(p for p in parts if p) or "Mixed"


def cluster_summary_labels(stats: pd.DataFrame) -> dict[int, str]:
    """Map cluster id → short regime label for use in charts."""
    labels: dict[int, str] = {}
    for cid in stats.index:
        row = stats.loc[cid]
        labels[int(cid)] = f"C{cid}: {regime_label(row)}"
    return labels
