from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.distribution import estimate_distribution
from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.regimes.multi_timeframe_regimes import MultiTimeframeRegimeModel
from rlm.types.forecast import ForecastConfig


def _is_datetime_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.DatetimeIndex)


def _resample_for_regime(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not _is_datetime_index(df):
        return df.copy()
    return df.resample(rule).last().dropna(how="all")


def _align_probs_to_index(
    probs: np.ndarray,
    src_index: pd.Index,
    dst_index: pd.Index,
    *,
    shift_for_safety: bool,
) -> np.ndarray:
    frame = pd.DataFrame(probs, index=src_index)
    if shift_for_safety and _is_datetime_index(frame):
        frame = frame.shift(1)
        frame = frame.bfill()
    aligned = frame.reindex(dst_index, method="ffill").bfill()
    return aligned.to_numpy(dtype=float)


class ForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
    ) -> None:
        self.config = config or ForecastConfig()
        self.move_window = move_window
        self.vol_window = vol_window

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = estimate_distribution(
            df=df,
            config=self.config,
            move_window=self.move_window,
            vol_window=self.vol_window,
        )
        out = compute_state_matrix_bands(out)
        return out


class HybridForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        hmm_config: HMMConfig | None = None,
        mtf_regimes: bool = False,
        mtf_htf_prob_paths: dict[str, str] | None = None,
        mtf_htf_weights: dict[str, float] | None = None,
        mtf_ltf_weight: float = 0.7,
        hierarchical: bool = False,
        macro_weight: float = 0.45,
        micro_timeframes: tuple[str, ...] = ("5min", "1min"),
        forecast_engine: object | None = None,
    ) -> None:
        """
        Initialize a hybrid forecasting pipeline by configuring its forecast engine, optional HMM, and optional multi-timeframe regime model.
        
        Parameters:
            config (ForecastConfig | None): Forecast configuration used when constructing the default ForecastPipeline; ignored if `forecast_engine` is provided.
            move_window (int): Lookback window length for the default ForecastPipeline.
            vol_window (int): Volatility window length for the default ForecastPipeline.
            hmm_config (HMMConfig | None): Configuration for an RLMHMM; when provided an HMM instance will be created and used.
            mtf_regimes (bool): When True, enable a MultiTimeframeRegimeModel for multi-timeframe regime annotation.
            mtf_htf_prob_paths (dict[str, str] | None): Mapping of high-timeframe probability column names to source paths for the MTF model.
            mtf_htf_weights (dict[str, float] | None): Weights for high-timeframe sources used by the MTF model.
            mtf_ltf_weight (float): Weight for the long-timeframe component in the MTF aggregation (clipped to [0, 1] where applied).
            hierarchical (bool): When True, enable hierarchical aggregation between macro and micro regime estimates.
            macro_weight (float): Weight applied to macro (longer timeframe) regime estimates when hierarchical aggregation is enabled; clipped to the [0.0, 1.0] range.
            micro_timeframes (tuple[str, ...]): Sequence of timeframe rules (e.g., "5min", "1min") used to build micro-level regime models.
            forecast_engine (object | None): Optional injected forecast engine instance to use instead of constructing the default ForecastPipeline.
        """
        if forecast_engine is not None:
            self.forecast = forecast_engine  # type: ignore[assignment]
        else:
            self.forecast = ForecastPipeline(
                config=config,
                move_window=move_window,
                vol_window=vol_window,
            )
        self.hmm = RLMHMM(hmm_config or HMMConfig()) if hmm_config else None
        self.mtf_regimes = bool(mtf_regimes)
        self.mtf = (
            MultiTimeframeRegimeModel(
                model="hmm",
                hmm_config=hmm_config or HMMConfig(),
                htf_prob_paths=mtf_htf_prob_paths or {},
                htf_weights=mtf_htf_weights or {},
                ltf_weight=mtf_ltf_weight,
            )
            if self.mtf_regimes
            else None
        )
        self.hierarchical = bool(hierarchical)
        self.macro_weight = float(np.clip(macro_weight, 0.0, 1.0))
        self.micro_timeframes = tuple(micro_timeframes)

    def run(self, df_features: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
        df = self.forecast.run(df_features)

        if self.hmm:
            if train_mask is not None:
                self.hmm.fit(df.loc[train_mask], verbose=False)
            else:
                self.hmm.fit(df, verbose=False)

            probs = self.hmm.predict_proba_filtered(df)
            if self.hierarchical:
                micro_sources: list[np.ndarray] = [probs]
                if _is_datetime_index(df):
                    for rule in self.micro_timeframes:
                        sampled = _resample_for_regime(df, rule)
                        if sampled.empty or len(sampled) >= len(df):
                            continue
                        sampled_mask = (
                            train_mask.reindex(sampled.index, fill_value=False)
                            if train_mask is not None
                            else None
                        )
                        micro_hmm = RLMHMM(deepcopy(self.hmm.config))
                        try:
                            micro_hmm.fit(
                                sampled.loc[sampled_mask] if sampled_mask is not None else sampled,
                                verbose=False,
                            )
                            sampled_probs = micro_hmm.predict_proba_filtered(sampled)
                            micro_sources.append(
                                _align_probs_to_index(
                                    sampled_probs,
                                    sampled.index,
                                    df.index,
                                    shift_for_safety=False,
                                )
                            )
                        except Exception:
                            continue
                micro_probs = np.mean(np.stack(micro_sources, axis=0), axis=0)

                if _is_datetime_index(df):
                    macro = _resample_for_regime(df, "1D")
                    macro_mask = (
                        train_mask.reindex(macro.index, fill_value=False)
                        if train_mask is not None
                        else None
                    )
                    if not macro.empty:
                        macro_hmm = RLMHMM(deepcopy(self.hmm.config))
                        try:
                            macro_hmm.fit(
                                macro.loc[macro_mask] if macro_mask is not None else macro,
                                verbose=False,
                            )
                            macro_probs = macro_hmm.predict_proba_filtered(macro)
                            macro_aligned = _align_probs_to_index(
                                macro_probs,
                                macro.index,
                                df.index,
                                shift_for_safety=True,
                            )
                            probs = (
                                self.macro_weight * macro_aligned
                                + (1.0 - self.macro_weight) * micro_probs
                            )
                            df["hmm_macro_probs"] = macro_aligned.tolist()
                            df["hmm_micro_probs"] = micro_probs.tolist()
                        except Exception:
                            probs = micro_probs
                    else:
                        probs = micro_probs
                else:
                    probs = micro_probs

            probs = np.clip(probs, 1e-12, None)
            probs = probs / probs.sum(axis=1, keepdims=True)
            df["hmm_probs"] = probs.tolist()
            df["hmm_state"] = np.argmax(probs, axis=1).astype(int)
            df["hmm_confidence"] = probs.max(axis=1).astype(float)
            if self.hmm.state_labels:
                df["hmm_state_label"] = [self.hmm.state_labels[int(s)] for s in df["hmm_state"]]

        if self.mtf is not None:
            self.mtf.fit(df.loc[train_mask] if train_mask is not None else df, verbose=False)
            df = self.mtf.annotate(df, prefix="mtf")

        return df


class HybridMarkovForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        markov_config: MarkovSwitchingConfig | None = None,
        model_path: str | None = None,
        hierarchical: bool = False,
        macro_weight: float = 0.45,
        micro_timeframes: tuple[str, ...] = ("5min", "1min"),
    ) -> None:
        self.forecast = ForecastPipeline(
            config=config,
            move_window=move_window,
            vol_window=vol_window,
        )
        self.markov = RLMMarkovSwitching(markov_config or MarkovSwitchingConfig())
        self.model_path = model_path
        self.hierarchical = bool(hierarchical)
        self.macro_weight = float(np.clip(macro_weight, 0.0, 1.0))
        self.micro_timeframes = tuple(micro_timeframes)

    def run(self, df_features: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
        df = self.forecast.run(df_features)
        self.markov.fit(df.loc[train_mask] if train_mask is not None else df, verbose=False)
        base_probs_df = self.markov.filter(df)
        probs = base_probs_df.to_numpy(dtype=float)
        micro_probs = probs.copy()

        if self.hierarchical and _is_datetime_index(df):
            micro_sources: list[np.ndarray] = [probs]
            for rule in self.micro_timeframes:
                sampled = _resample_for_regime(df, rule)
                if sampled.empty or len(sampled) >= len(df):
                    continue
                sampled_mask = (
                    train_mask.reindex(sampled.index, fill_value=False)
                    if train_mask is not None
                    else None
                )
                micro_model = RLMMarkovSwitching(deepcopy(self.markov.config))
                try:
                    micro_model.fit(
                        sampled.loc[sampled_mask] if sampled_mask is not None else sampled,
                        verbose=False,
                    )
                    sampled_probs = micro_model.filter(sampled).to_numpy(dtype=float)
                    micro_sources.append(
                        _align_probs_to_index(
                            sampled_probs,
                            sampled.index,
                            df.index,
                            shift_for_safety=False,
                        )
                    )
                except Exception:
                    continue
            micro_probs = np.mean(np.stack(micro_sources, axis=0), axis=0)
            macro = _resample_for_regime(df, "1D")
            if not macro.empty:
                macro_mask = (
                    train_mask.reindex(macro.index, fill_value=False)
                    if train_mask is not None
                    else None
                )
                macro_model = RLMMarkovSwitching(deepcopy(self.markov.config))
                try:
                    macro_model.fit(
                        macro.loc[macro_mask] if macro_mask is not None else macro, verbose=False
                    )
                    macro_probs = macro_model.filter(macro).to_numpy(dtype=float)
                    macro_aligned = _align_probs_to_index(
                        macro_probs,
                        macro.index,
                        df.index,
                        shift_for_safety=True,
                    )
                    probs = (
                        self.macro_weight * macro_aligned + (1.0 - self.macro_weight) * micro_probs
                    )
                except Exception:
                    probs = micro_probs
            else:
                probs = micro_probs

        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        out = df.copy()
        if self.hierarchical:
            out["markov_micro_probs"] = micro_probs.tolist()
            if _is_datetime_index(df):
                out["markov_macro_probs"] = (
                    np.full_like(micro_probs, np.nan)
                    if "macro_aligned" not in locals()
                    else macro_aligned
                ).tolist()
        out["markov_probs"] = probs.tolist()
        out["markov_state"] = np.argmax(probs, axis=1).astype(int)
        out["markov_confidence"] = probs.max(axis=1).astype(float)
        if self.markov.state_labels:
            out["markov_state_label"] = [
                self.markov.state_labels[int(s)] for s in out["markov_state"]
            ]
        return out


class HybridProbabilisticForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        hmm_config: HMMConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self.forecast = ProbabilisticForecastPipeline(
            config=config,
            move_window=move_window,
            vol_window=vol_window,
            model_path=model_path,
        )
        self.hmm = RLMHMM(hmm_config or HMMConfig()) if hmm_config else None

    def run(self, df_features: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
        df = self.forecast.run(df_features)

        if self.hmm:
            if train_mask is not None:
                self.hmm.fit(df.loc[train_mask], verbose=False)
            else:
                self.hmm.fit(df, verbose=False)

            probs = self.hmm.predict_proba_filtered(df)
            df["hmm_probs"] = probs.tolist()
            df["hmm_state"] = self.hmm.most_likely_state_filtered(df)
            if self.hmm.state_labels:
                df["hmm_state_label"] = [self.hmm.state_labels[int(s)] for s in df["hmm_state"]]

        return df


class HybridMarkovProbabilisticForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        markov_config: MarkovSwitchingConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self.forecast = ProbabilisticForecastPipeline(
            config=config,
            move_window=move_window,
            vol_window=vol_window,
            model_path=model_path,
        )
        self.markov = RLMMarkovSwitching(markov_config or MarkovSwitchingConfig())

    def run(self, df_features: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
        df = self.forecast.run(df_features)
        self.markov.fit(df.loc[train_mask] if train_mask is not None else df, verbose=False)
        return self.markov.annotate(df, prefix="markov")
