"""
Torch Kronos model entrypoints.

The upstream Kronos decoder weights and tokenizer live on HuggingFace; this repository
does not vendor the full ``KronosPredictor.predict`` implementation.  Production installs
that need autoregressive inference should embed the reference implementation (see
https://github.com/DGator86/Kronos) or extend these stubs.

Unit tests inject ``predict_paths`` mocks via :class:`~rlm.forecasting.kronos_forecast.KronosForecastPipeline`
and :class:`~rlm.forecasting.models.kronos.predictor.RLMKronosPredictor`.
"""

from __future__ import annotations


class KronosTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> KronosTokenizer:
        raise ImportError(
            "KronosTokenizer.from_pretrained is not available in the stub module. "
            "Vendor Kronos model code or use KronosForecastPipeline(predictor=mock) in tests."
        )


class Kronos:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> Kronos:
        raise ImportError(
            "Kronos.from_pretrained is not available in the stub module. "
            "Vendor Kronos model code or use KronosForecastPipeline(predictor=mock) in tests."
        )


class KronosPredictor:
    """Torch predictor with ``predict(df, x_timestamp, y_timestamp, ...) -> DataFrame``."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise ImportError(
            "KronosPredictor is a stub unless you vendor the full Kronos inference stack."
        )

    def predict(self, **_kwargs):  # pragma: no cover - stub
        raise ImportError("KronosPredictor.predict requires the vendored Kronos implementation.")
