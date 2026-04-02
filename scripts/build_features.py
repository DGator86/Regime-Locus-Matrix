from __future__ import annotations

import pandas as pd

from rlm.factors.pipeline import FactorPipeline


def main() -> None:
    df = pd.read_csv("data/raw/sample_bars.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    pipeline = FactorPipeline()
    features = pipeline.run(df)

    features.to_csv("data/processed/features.csv")
    print(features.tail(5)[["S_D", "S_V", "S_L", "S_G"]])


if __name__ == "__main__":
    main()
