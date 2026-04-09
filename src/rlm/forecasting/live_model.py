from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
)
from rlm.roee.pipeline import ROEEConfig
from rlm.types.forecast import ForecastConfig

RegimeModelName = Literal["forecast", "hmm", "markov"]


class LiveForecastParameters(BaseModel):
    drift_gamma_alpha: float = 0.65
    sigma_floor: float = 1e-4
    direction_neutral_threshold: float = 0.3
    move_window: int = 100
    vol_window: int = 100

    def to_forecast_config(self) -> ForecastConfig:
        return ForecastConfig(
            drift_gamma_alpha=self.drift_gamma_alpha,
            sigma_floor=self.sigma_floor,
            direction_neutral_threshold=self.direction_neutral_threshold,
        )


class LiveROEEParameters(BaseModel):
    confidence_threshold: float = 0.6
    sizing_multiplier: float = 1.0
    transition_penalty: float = 0.5
    use_dynamic_sizing: bool = False
    vol_target: float = 0.15
    max_kelly_fraction: float = 0.25
    max_capital_fraction: float = 0.5

    def to_roee_config(self) -> ROEEConfig:
        return ROEEConfig(
            hmm_confidence_threshold=self.confidence_threshold,
            sizing_multiplier=self.sizing_multiplier,
            transition_penalty=self.transition_penalty,
            use_dynamic_sizing=self.use_dynamic_sizing,
            vol_target=self.vol_target,
            max_kelly_fraction=self.max_kelly_fraction,
            max_capital_fraction=self.max_capital_fraction,
        )

    def decision_kwargs(self) -> dict[str, float | bool]:
        return {
            "hmm_confidence_threshold": self.confidence_threshold,
            "hmm_sizing_multiplier": self.sizing_multiplier,
            "hmm_transition_penalty": self.transition_penalty,
            "use_dynamic_sizing": self.use_dynamic_sizing,
            "vol_target": self.vol_target,
            "max_kelly_fraction": self.max_kelly_fraction,
            "max_capital_fraction": self.max_capital_fraction,
        }


class LiveHMMParameters(BaseModel):
    n_states: int = 6
    n_iter: int = 100
    filter_backend: Literal["auto", "numpy", "numba"] = "auto"
    prefer_gpu: bool = False
    hierarchical: bool = True
    macro_weight: float = 0.45
    micro_timeframes: tuple[str, ...] = ("5min", "1min")

    def to_hmm_config(self) -> HMMConfig:
        return HMMConfig(
            n_states=self.n_states,
            n_iter=self.n_iter,
            filter_backend=self.filter_backend,
            prefer_gpu=self.prefer_gpu,
        )


class LiveMarkovParameters(BaseModel):
    n_states: int = 3
    switching_variance: bool = True
    trend: str = "c"
    hierarchical: bool = True
    macro_weight: float = 0.45
    micro_timeframes: tuple[str, ...] = ("5min", "1min")

    def to_markov_config(self) -> MarkovSwitchingConfig:
        return MarkovSwitchingConfig(
            n_states=self.n_states,
            switching_variance=self.switching_variance,
            trend=self.trend,
        )


class LiveRegimeModelConfig(BaseModel):
    model: RegimeModelName = "forecast"
    forecast: LiveForecastParameters = Field(default_factory=LiveForecastParameters)
    roee: LiveROEEParameters = Field(default_factory=LiveROEEParameters)
    hmm: LiveHMMParameters = Field(default_factory=LiveHMMParameters)
    markov: LiveMarkovParameters = Field(default_factory=LiveMarkovParameters)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def build_pipeline(
        self,
    ) -> ForecastPipeline | HybridForecastPipeline | HybridMarkovForecastPipeline:
        forecast_config = self.forecast.to_forecast_config()
        if self.model == "hmm":
            return HybridForecastPipeline(
                config=forecast_config,
                move_window=self.forecast.move_window,
                vol_window=self.forecast.vol_window,
                hmm_config=self.hmm.to_hmm_config(),
                hierarchical=self.hmm.hierarchical,
                macro_weight=self.hmm.macro_weight,
                micro_timeframes=self.hmm.micro_timeframes,
            )
        if self.model == "markov":
            return HybridMarkovForecastPipeline(
                config=forecast_config,
                move_window=self.forecast.move_window,
                vol_window=self.forecast.vol_window,
                markov_config=self.markov.to_markov_config(),
                hierarchical=self.markov.hierarchical,
                macro_weight=self.markov.macro_weight,
                micro_timeframes=self.markov.micro_timeframes,
            )
        return ForecastPipeline(
            config=forecast_config,
            move_window=self.forecast.move_window,
            vol_window=self.forecast.vol_window,
        )

    def decision_kwargs(self) -> dict[str, float | bool]:
        return self.roee.decision_kwargs()


def load_live_regime_model(path: Path) -> LiveRegimeModelConfig:
    return LiveRegimeModelConfig.model_validate_json(path.read_text(encoding="utf-8"))


def save_live_regime_model(config: LiveRegimeModelConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
