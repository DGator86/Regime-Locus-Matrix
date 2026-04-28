#!/usr/bin/env python3
"""Fit isotonic calibration for one-step-ahead regime top-1 hit rate.

Reads existing forecast feature CSVs (must contain transition columns from a recent
pipeline).  Builds supervised pairs::

    p_hat[t]   = *_most_likely_next_prob[t]
    pred[t]    = *_most_likely_next_state[t]
    actual[t]  = *_state[t+1]

Fits ``E[hit | p_hat]`` with a monotone isotonic map (requires scikit-learn) and writes
``data/processed/regime_transition_calibration.json``.  At runtime, set
``RLM_TRANSITION_CALIBRATION`` or rely on the default path; the family in the JSON must
match ``hmm`` or ``markov``.

Examples::

    python scripts/fit_regime_transition_calibration.py --symbol SPY --regime-family hmm
    python scripts/fit_regime_transition_calibration.py --universe --regime-family hmm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm.cli.common import normalize_symbol  # noqa: E402
from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE  # noqa: E402
from rlm.data.paths import get_data_root, get_processed_data_dir  # noqa: E402
from rlm.regimes.transition_calibration import (  # noqa: E402
    default_calibration_path,
    fit_isotonic_top1_next_regime,
    save_calibration,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY", help="Single ticker")
    p.add_argument("--universe", action="store_true", help="Use EXPANDED_LIQUID_UNIVERSE")
    p.add_argument(
        "--regime-family",
        choices=("hmm", "markov"),
        default="hmm",
        help="Which columns to read from forecast CSVs",
    )
    p.add_argument("--data-root", default=None, help="RLM data root (default: env / ./data)")
    p.add_argument("--out", default=None, help="Output JSON path (default: processed/regime_transition_calibration.json)")
    p.add_argument("--min-rows", type=int, default=80, help="Minimum aligned rows per symbol before merging")
    return p.parse_args()


def _load_pairs_from_forecast_csv(
    path: Path,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except OSError:
        return None
    pcol = f"{family}_most_likely_next_prob"
    pred_col = f"{family}_most_likely_next_state"
    act_col = f"{family}_state"
    if pcol not in df.columns or pred_col not in df.columns or act_col not in df.columns:
        return None
    p_hat = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(df[act_col], errors="coerce").to_numpy(dtype=float)
    if len(p_hat) < 3:
        return None
    p_hat_t = p_hat[:-1]
    pred_t = pred[:-1]
    actual_tp1 = actual[1:]
    return p_hat_t, pred_t.astype(int), actual_tp1.astype(int)


def main() -> int:
    args = _parse_args()
    dr = get_data_root(args.data_root)
    proc = get_processed_data_dir(dr)
    symbols = list(EXPANDED_LIQUID_UNIVERSE) if args.universe else [normalize_symbol(args.symbol)]

    all_p: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    used: list[str] = []

    for sym in symbols:
        path = proc / f"forecast_features_{sym}.csv"
        triplet = _load_pairs_from_forecast_csv(path, args.regime_family)
        if triplet is None:
            continue
        p_hat, pred, actual = triplet
        if len(p_hat) < args.min_rows:
            continue
        all_p.append(p_hat)
        all_pred.append(pred)
        all_act.append(actual)
        used.append(sym)

    if not all_p:
        print(
            "No usable forecast CSVs with transition columns. "
            f"Expected files like {proc / 'forecast_features_SPY.csv'} "
            f"with {args.regime_family}_most_likely_next_prob, "
            f"{args.regime_family}_most_likely_next_state, {args.regime_family}_state.",
            file=sys.stderr,
        )
        return 1

    p_hat = np.concatenate(all_p)
    pred = np.concatenate(all_pred)
    actual = np.concatenate(all_act)
    payload = fit_isotonic_top1_next_regime(
        p_hat,
        pred,
        actual,
        regime_family=args.regime_family,
    )
    if payload is None:
        print(
            "Calibration fit failed (need scikit-learn and enough samples). "
            "Install: pip install scikit-learn",
            file=sys.stderr,
        )
        return 1
    payload["symbols_used"] = used
    out_path = Path(args.out).expanduser().resolve() if args.out else default_calibration_path(dr)
    save_calibration(payload, out_path)
    print(f"Wrote {out_path} (n={payload['n_samples']} symbols={len(used)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
