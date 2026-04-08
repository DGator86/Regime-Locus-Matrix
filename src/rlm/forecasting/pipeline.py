from __future__ import annotations

import pandas as pd

from rlm.forecasting.bands import compute_state_matrix_bands
from rlm.forecasting.distribution import estimate_distribution
from rlm.forecasting.hmm import HMMConfig, RLMHMM
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.types.forecast import ForecastConfig


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
    ) -> None:
        self.forecast = ForecastPipeline(
            config=config,
            move_window=move_window,
            vol_window=vol_window,
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


class HybridMarkovForecastPipeline:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        move_window: int = 100,
        vol_window: int = 100,
        markov_config: MarkovSwitchingConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self.forecast = ForecastPipeline(
            config=config,
            move_window=move_window,
            vol_window=vol_window,
        )
        self.markov = RLMMarkovSwitching(markov_config or MarkovSwitchingConfig())
        self.model_path = model_path

    def run(self, df_features: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
        df = self.forecast.run(df_features)
        self.markov.fit(df.loc[train_mask] if train_mask is not None else df, verbose=False)
        return self.markov.annotate(df, prefix="markov")


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
