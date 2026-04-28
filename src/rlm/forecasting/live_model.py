from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.kronos_forecast import KronosBlendPipeline, KronosConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.engines import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
)
from rlm.roee.engine import ROEEConfig
from rlm.types.forecast import ForecastConfig

RegimeModelName = Literal["forecast", "hmm", "markov"]


class LiveKronosParameters(BaseModel):
    """Kronos blend parameters stored inside the live regime model config."""

    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str | None = None
    max_context: int = 512
    lookback: int = 200
    pred_len: int = 1
    sample_count: int = 5
    stride: int = 1
    weight: float = 0.35  # blend weight: 0 = base only, 1 = Kronos only

    def to_kronos_config(self) -> KronosConfig:
        return KronosConfig(
            model_name=self.model_name,
            tokenizer_name=self.tokenizer_name,
            device=self.device,
            max_context=self.max_context,
            lookback=self.lookback,
            pred_len=self.pred_len,
            sample_count=self.sample_count,
            stride=self.stride,
        )


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
    high_vol_kelly_multiplier: float = 0.5
    transition_kelly_multiplier: float = 0.75
    calm_trend_kelly_multiplier: float = 1.25

    def to_roee_config(self) -> ROEEConfig:
        return ROEEConfig(
            hmm_confidence_threshold=self.confidence_threshold,
            sizing_multiplier=self.sizing_multiplier,
            transition_penalty=self.transition_penalty,
            use_dynamic_sizing=self.use_dynamic_sizing,
            vol_target=self.vol_target,
            max_kelly_fraction=self.max_kelly_fraction,
            max_capital_fraction=self.max_capital_fraction,
            high_vol_kelly_multiplier=self.high_vol_kelly_multiplier,
            transition_kelly_multiplier=self.transition_kelly_multiplier,
            calm_trend_kelly_multiplier=self.calm_trend_kelly_multiplier,
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
            "high_vol_kelly_multiplier": self.high_vol_kelly_multiplier,
            "transition_kelly_multiplier": self.transition_kelly_multiplier,
            "calm_trend_kelly_multiplier": self.calm_trend_kelly_multiplier,
        }


class LiveHMMParameters(BaseModel):
    n_states: int = 6
    n_iter: int = 100
    filter_backend: Literal["auto", "numpy", "numba"] = "auto"
    prefer_gpu: bool = False
    transition_pseudocount: float = 0.1
    hierarchical: bool = True
    macro_weight: float = 0.45
    micro_timeframes: tuple[str, ...] = ("5min", "1min")

    def to_hmm_config(self) -> HMMConfig:
        return HMMConfig(
            n_states=self.n_states,
            n_iter=self.n_iter,
            filter_backend=self.filter_backend,
            prefer_gpu=self.prefer_gpu,
            transition_pseudocount=self.transition_pseudocount,
        )


class LiveMarkovParameters(BaseModel):
    n_states: int = 3
    switching_variance: bool = True
    trend: str = "c"
    transition_pseudocount: float = 0.1
    hierarchical: bool = True
    macro_weight: float = 0.45
    micro_timeframes: tuple[str, ...] = ("5min", "1min")

    def to_markov_config(self) -> MarkovSwitchingConfig:
        return MarkovSwitchingConfig(
            n_states=self.n_states,
            switching_variance=self.switching_variance,
            trend=self.trend,
            transition_pseudocount=self.transition_pseudocount,
        )


class LiveRegimeModelConfig(BaseModel):
    model: RegimeModelName = "forecast"
    forecast: LiveForecastParameters = Field(default_factory=LiveForecastParameters)
    roee: LiveROEEParameters = Field(default_factory=LiveROEEParameters)
    hmm: LiveHMMParameters = Field(default_factory=LiveHMMParameters)
    markov: LiveMarkovParameters = Field(default_factory=LiveMarkovParameters)
    use_kronos: bool = False
    kronos: LiveKronosParameters = Field(default_factory=LiveKronosParameters)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def build_pipeline(
        self,
    ) -> ForecastPipeline | HybridForecastPipeline | HybridMarkovForecastPipeline | KronosBlendPipeline:
        """Build the active forecast pipeline, optionally wrapped in a Kronos blend layer.

        When ``use_kronos=True`` the returned pipeline is a ``KronosBlendPipeline``
        that transparently runs the base model first and then blends in the Kronos
        deep-learning forecast at ``kronos.weight``.
        """
        forecast_config = self.forecast.to_forecast_config()
        if self.model == "hmm":
            base: ForecastPipeline | HybridForecastPipeline | HybridMarkovForecastPipeline = (
                HybridForecastPipeline(
                    config=forecast_config,
                    move_window=self.forecast.move_window,
                    vol_window=self.forecast.vol_window,
                    hmm_config=self.hmm.to_hmm_config(),
                    hierarchical=self.hmm.hierarchical,
                    macro_weight=self.hmm.macro_weight,
                    micro_timeframes=self.hmm.micro_timeframes,
                )
            )
        elif self.model == "markov":
            base = HybridMarkovForecastPipeline(
                config=forecast_config,
                move_window=self.forecast.move_window,
                vol_window=self.forecast.vol_window,
                markov_config=self.markov.to_markov_config(),
                hierarchical=self.markov.hierarchical,
                macro_weight=self.markov.macro_weight,
                micro_timeframes=self.markov.micro_timeframes,
            )
        else:
            base = ForecastPipeline(
                config=forecast_config,
                move_window=self.forecast.move_window,
                vol_window=self.forecast.vol_window,
            )
        if self.use_kronos:
            return KronosBlendPipeline(
                base_pipeline=base,
                kronos_config=self.kronos.to_kronos_config(),
                weight=self.kronos.weight,
            )
        return base

    def decision_kwargs(self) -> dict[str, float | bool]:
        return self.roee.decision_kwargs()


def apply_nightly_hyperparam_overlay(cfg: LiveRegimeModelConfig, repo_root: Path) -> LiveRegimeModelConfig:
    """Merge ``data/processed/live_nightly_hyperparams.json`` into a live config when present.

    Keys produced by :class:`rlm.optimization.nightly.NightlyMTFOptimizer` are mapped onto
    :class:`LiveForecastParameters` / :class:`LiveROEEParameters` where names align; unknown keys
    are ignored so the live stack stays forward-compatible.
    """
    path = repo_root / "data" / "processed" / "live_nightly_hyperparams.json"
    if not path.is_file():
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    if not isinstance(raw, dict) or not raw:
        return cfg
    d = cfg.model_dump()
    fo: dict[str, Any] = d["forecast"]
    rw: dict[str, Any] = d["roee"]
    if (v := raw.get("move_window")) is not None:
        fo["move_window"] = int(v)
    if (v := raw.get("vol_window")) is not None:
        fo["vol_window"] = int(v)
    if (v := raw.get("direction_neutral_threshold")) is not None:
        fo["direction_neutral_threshold"] = float(v)
    if (v := raw.get("hmm_confidence_threshold")) is not None:
        rw["confidence_threshold"] = float(v)
    if (v := raw.get("high_vol_kelly_multiplier")) is not None:
        rw["high_vol_kelly_multiplier"] = float(v)
    if (v := raw.get("transition_kelly_multiplier")) is not None:
        rw["transition_kelly_multiplier"] = float(v)
    if (v := raw.get("calm_trend_kelly_multiplier")) is not None:
        rw["calm_trend_kelly_multiplier"] = float(v)
    d["forecast"] = fo
    d["roee"] = rw
    return LiveRegimeModelConfig.model_validate(d)


def load_live_regime_model(path: Path) -> LiveRegimeModelConfig:
    return LiveRegimeModelConfig.model_validate_json(path.read_text(encoding="utf-8"))


def save_live_regime_model(config: LiveRegimeModelConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
