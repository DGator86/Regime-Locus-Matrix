#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd

from rlm.roee.engine import apply_roee_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Permutation importance for ROEE policy outputs")
    p.add_argument("--forecast-csv", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--prefixes", nargs="*", default=["std_", "raw_", "hmm_", "kronos_", "S_"],
                   help="Feature prefixes to include in permutation candidates.")
    args = p.parse_args()

    df = pd.read_csv(args.forecast_csv)
    base = apply_roee_policy(df)
    baseline = base["roee_size_fraction"].fillna(0.0)
    candidates = [c for c in df.columns if any(c.startswith(pref) for pref in args.prefixes)]

    rows = []
    for c in candidates:
        perm = df.copy()
        perm[c] = perm[c].sample(frac=1.0, random_state=42).to_numpy()
        out = apply_roee_policy(perm)
        delta = (baseline - out["roee_size_fraction"].fillna(0.0)).abs().mean()
        rows.append({"feature": c, "importance": float(delta)})

    imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
    print(imp.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
