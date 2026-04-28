from __future__ import annotations

import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class DealerFlowFactors(FactorCalculator):
    """
    Expected columns where available:
      gex, vanna, charm, put_call_skew, iv_rank,
      term_structure_ratio, dealer_position_proxy
    """

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="gex_signal",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="vanna_signal",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="charm_signal",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="put_call_skew",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.03,
                k=1.0,
                invert=True,
            ),
            FactorSpec(
                name="iv_rank_ratio",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.5,
                k=0.8,
            ),
            FactorSpec(
                name="term_structure_ratio",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="dealer_position_proxy",
                category=FactorCategory.DEALER_FLOW,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.10,
                k=0.9,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        out["gex_signal"] = df["gex"] if "gex" in df.columns else pd.NA
        out["vanna_signal"] = df["vanna"] if "vanna" in df.columns else pd.NA
        out["charm_signal"] = df["charm"] if "charm" in df.columns else pd.NA
        out["put_call_skew"] = df["put_call_skew"] if "put_call_skew" in df.columns else pd.NA
        out["iv_rank_ratio"] = df["iv_rank"] if "iv_rank" in df.columns else pd.NA
        out["term_structure_ratio"] = df["term_structure_ratio"] if "term_structure_ratio" in df.columns else pd.NA
        out["dealer_position_proxy"] = df["dealer_position_proxy"] if "dealer_position_proxy" in df.columns else pd.NA

        return out
