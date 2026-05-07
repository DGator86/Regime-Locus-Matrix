"""Probabilistic Regime Engine (PRE) — continuous-confidence replacement for the binary gate.

Implements the mathematical specification:

  1. Filtered state probabilities (data certainty backbone)
  2. Dynamic per-state attractiveness g(i) via exceedance probability
  3. Horizon-averaged forward-looking confidence (transition momentum)
  4. Optional Bayesian Kronos update (sensor fusion, no binary comparison)
  5. Multi-timeframe HTF+LTF conditioning via joint belief product

The engine outputs C_t ∈ [0,1] — a single continuously-varying score used by
ROEE as a direct size multiplier:  allocation = C_t * kelly_fraction * capital.

All model parameters are computed strictly on lagged data (causal walkforward).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.special import ndtr  # standard normal CDF, Φ(x) = ndtr(x)
from scipy.stats import norm

from rlm.forecasting.hmm import RLMHMM, HMMConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PREConfig(BaseModel):
    """Configuration for both single-TF and MTF probabilistic regime engines."""

    k_l: int = Field(6, ge=2, le=15, description="Number of LTF hidden states.")
    k_h: int = Field(3, ge=2, le=10, description="Number of HTF hidden states (MTF only).")
    horizon: int = Field(5, ge=1, le=60, description="Look-ahead horizon H for the confidence average.")
    r_min: float = Field(0.0, description="Minimum acceptable daily return threshold for g(i).")
    kronos_enabled: bool = True
    htf_resample_rule: str = Field(
        "W-FRI",
        description="Pandas resample rule for HTF aggregation (default: weekly, Friday close).",
    )
    returns_col: str = "close"
    min_state_samples: int = Field(
        5,
        ge=0,
        description="Minimum observations per state to compute valid g(i); states below this use g=0.5.",
    )
    kronos_min_state_samples: int = Field(
        3,
        ge=0,
        description="Minimum Kronos observations per state to fit conditional distribution.",
    )
    hmm_config: HMMConfig = Field(default_factory=HMMConfig)
    htf_hmm_config: HMMConfig = Field(default_factory=lambda: HMMConfig(n_states=3))


# ---------------------------------------------------------------------------
# Output signal
# ---------------------------------------------------------------------------


@dataclass
class RegimeSignal:
    """Structured output from a single inference step of the PRE."""

    confidence: float
    """Final horizon-averaged trade confidence C_t ∈ [0,1]. Feed directly to ROEE sizing."""

    ltf_belief_raw: np.ndarray
    """LTF filtered state probabilities *before* Kronos update, shape (K_L,)."""

    ltf_belief_post_kronos: np.ndarray
    """LTF state probabilities *after* Bayesian Kronos update, shape (K_L,).
    Equal to ``ltf_belief_raw`` when Kronos is disabled."""

    expected_attractiveness_path: np.ndarray
    """Per-step expected attractiveness for k=0..H-1, shape (H,). Useful for diagnostics."""

    instantaneous_attractiveness: float
    """Spot suitability s_spot = α' · g at k=0, before horizon-averaging."""

    current_most_likely_ltf_state: int
    """Argmax of ``ltf_belief_post_kronos``."""

    htf_belief: Optional[np.ndarray] = None
    """HTF state probabilities, shape (K_H,). None when MTF is disabled."""

    joint_belief: Optional[np.ndarray] = None
    """Flattened joint belief ψ(h,l) = β(h)·α(l), shape (K_H*K_L,). None when MTF is disabled."""

    current_most_likely_htf_state: Optional[int] = None
    """Argmax of ``htf_belief``. None when MTF is disabled."""


# ---------------------------------------------------------------------------
# Pure numerical helpers
# ---------------------------------------------------------------------------


def _compute_attractiveness(
    returns_by_state: dict[int, np.ndarray],
    k: int,
    r_min: float,
    min_samples: int,
) -> np.ndarray:
    """Compute g(i) = P(r > r_min | state=i) = 1 - Φ((r_min - μ_i) / σ_i) for all i.

    States with too few samples default to g=0.5 (maximum uncertainty).
    """
    g = np.full(k, 0.5, dtype=np.float64)
    for state in range(k):
        rets = returns_by_state.get(state, np.array([]))
        if len(rets) < min_samples:
            continue
        mu = float(np.mean(rets))
        sigma = float(np.std(rets))
        if sigma < 1e-12:
            g[state] = 1.0 if mu > r_min else 0.0
        else:
            g[state] = float(1.0 - ndtr((r_min - mu) / sigma))
    return np.clip(g, 0.0, 1.0)


def _horizon_averaged_score(
    posterior: np.ndarray,
    transmat: np.ndarray,
    attractiveness: np.ndarray,
    horizon: int,
) -> tuple[float, np.ndarray]:
    """Compute C_t = (1/H) * Σ_{k=0}^{H-1} posterior^T @ T^k @ g.

    Returns (C_t, path) where path[k] = E[g(S_{t+k})] for diagnostics.
    """
    K = len(attractiveness)
    if posterior.shape[0] != K or transmat.shape != (K, K):
        raise ValueError(
            f"Shape mismatch: posterior={posterior.shape}, transmat={transmat.shape}, g={attractiveness.shape}"
        )
    path = np.zeros(horizon, dtype=np.float64)
    p = posterior.copy().astype(np.float64)
    for k in range(horizon):
        path[k] = float(p @ attractiveness)
        if k < horizon - 1:
            p = p @ transmat
            # keep numerically valid
            p = np.clip(p, 1e-12, None)
            p /= p.sum()
    return float(path.mean()), path


def _bayesian_kronos_update(
    prior: np.ndarray,
    kronos_forecast: float,
    kronos_means: np.ndarray,
    kronos_stds: np.ndarray,
) -> np.ndarray:
    """Update state belief with Kronos forecast as an independent observation.

    Likelihood: p(f | state=i) = N(f; ν_i, τ_i²).
    Returns normalised posterior probability vector.
    """
    likelihoods = np.array(
        [float(norm.pdf(kronos_forecast, loc=mu, scale=max(sd, 1e-8))) for mu, sd in zip(kronos_means, kronos_stds)],
        dtype=np.float64,
    )
    likelihoods = np.clip(likelihoods, 1e-300, None)
    posterior = prior * likelihoods
    total = posterior.sum()
    if total < 1e-300:
        return prior.copy()
    return posterior / total


# ---------------------------------------------------------------------------
# Fitted artefact containers (lightweight, serialisable)
# ---------------------------------------------------------------------------


@dataclass
class _PRESingleTFArtefacts:
    """All causal artefacts from a single-TF PRE fit(), keyed by training cut-off."""

    hmm: RLMHMM
    transmat: np.ndarray          # (K, K) calibrated
    attractiveness: np.ndarray    # (K,)   g(i)
    kronos_means: np.ndarray      # (K,)   ν_i
    kronos_stds: np.ndarray       # (K,)   τ_i
    model_timestamp: str = ""


@dataclass
class _PREMTFArtefacts:
    """Artefacts from a full MTF fit (HTF + LTF layers)."""

    htf: _PRESingleTFArtefacts
    ltf: _PRESingleTFArtefacts
    model_timestamp: str = ""


# ---------------------------------------------------------------------------
# Single-timeframe Probabilistic Regime Engine
# ---------------------------------------------------------------------------


class ProbabilisticRegimeEngine:
    """Continuous-confidence regime engine for a single timeframe.

    Replaces the binary ``allowed_to_trade`` gate with a smooth C_t ∈ [0,1]
    that combines filtered state probabilities, dynamic state attractiveness,
    transition momentum, and an optional Bayesian Kronos sensor fusion.

    Walkforward usage
    -----------------
    Fit on expanding in-sample windows; call ``score()`` for each live bar.

    >>> engine = ProbabilisticRegimeEngine(config)
    >>> engine.fit(df_train)
    >>> signal = engine.score(latest_filtered_probs, kronos_forecast=0.003)
    >>> allocation = signal.confidence * kelly_fraction * capital
    """

    def __init__(self, config: PREConfig | None = None) -> None:
        self.config = config or PREConfig()
        self._artefacts: _PRESingleTFArtefacts | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        kronos_col: str | None = "kronos_forecast",
        train_timestamp: str = "",
    ) -> "ProbabilisticRegimeEngine":
        """Fit HMM and derive all causal artefacts from *df_train*.

        Parameters
        ----------
        df_train:
            Feature DataFrame containing the columns expected by RLMHMM
            (``S_D``, ``S_V``, ``S_L``, ``S_G``) plus ``returns_col`` for
            return-based attractiveness and optionally ``kronos_col``.
        kronos_col:
            Column with Kronos forecasts for conditional distribution fitting.
            Pass ``None`` to skip Kronos fitting even if the column exists.
        train_timestamp:
            ISO timestamp string stamping the training cut-off (for logging).
        """
        cfg = self.config
        hmm_cfg = cfg.hmm_config.model_copy(update={"n_states": cfg.k_l})
        rlm_hmm = RLMHMM(hmm_cfg)
        rlm_hmm.fit(df_train, verbose=False)
        transmat = rlm_hmm.calibrated_transmat()

        # --- per-state return statistics ---------------------------------
        probs_filtered = rlm_hmm.predict_proba_filtered(df_train)
        state_assignments = np.argmax(probs_filtered, axis=1)
        returns_by_state = self._collect_state_returns(df_train, state_assignments, cfg.k_l)
        g = _compute_attractiveness(returns_by_state, cfg.k_l, cfg.r_min, cfg.min_state_samples)

        # --- Kronos conditional distributions ----------------------------
        kronos_means = np.zeros(cfg.k_l, dtype=np.float64)
        kronos_stds = np.ones(cfg.k_l, dtype=np.float64)
        if cfg.kronos_enabled and kronos_col and kronos_col in df_train.columns:
            kronos_vals = pd.to_numeric(df_train[kronos_col], errors="coerce")
            for s in range(cfg.k_l):
                mask = state_assignments == s
                vals = kronos_vals[mask].dropna().values
                if len(vals) >= cfg.kronos_min_state_samples:
                    kronos_means[s] = float(np.mean(vals))
                    kronos_stds[s] = float(max(np.std(vals), 1e-8))

        self._artefacts = _PRESingleTFArtefacts(
            hmm=rlm_hmm,
            transmat=transmat,
            attractiveness=g,
            kronos_means=kronos_means,
            kronos_stds=kronos_stds,
            model_timestamp=train_timestamp,
        )
        return self

    def _collect_state_returns(
        self,
        df: pd.DataFrame,
        state_assignments: np.ndarray,
        k: int,
    ) -> dict[int, np.ndarray]:
        returns_col = self.config.returns_col
        if returns_col in df.columns:
            price = pd.to_numeric(df[returns_col], errors="coerce")
            rets = price.pct_change().fillna(0.0).values
        else:
            log.warning("PRE: returns column %r not found; g(i) will default to 0.5.", returns_col)
            rets = np.zeros(len(df))
        result: dict[int, np.ndarray] = {}
        for s in range(k):
            mask = state_assignments == s
            result[s] = rets[mask].astype(np.float64)
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(
        self,
        filtered_probs: np.ndarray,
        kronos_forecast: float | None = None,
    ) -> RegimeSignal:
        """Compute the confidence score C_t for a single bar.

        Parameters
        ----------
        filtered_probs:
            Forward-filtered state probabilities for the current bar, shape (K,).
            Obtain via ``RLMHMM.predict_proba_filtered(df)[-1]``.
        kronos_forecast:
            Scalar Kronos forecast for the current period (optional).
            When provided and ``config.kronos_enabled``, performs a Bayesian
            update before the horizon-averaged score.
        """
        if self._artefacts is None:
            raise RuntimeError("ProbabilisticRegimeEngine must be fitted before scoring.")
        arts = self._artefacts
        alpha = np.array(filtered_probs, dtype=np.float64)
        alpha = np.clip(alpha, 1e-12, None)
        alpha /= alpha.sum()

        # Bayesian Kronos update
        if self.config.kronos_enabled and kronos_forecast is not None and np.isfinite(kronos_forecast):
            posterior = _bayesian_kronos_update(alpha, kronos_forecast, arts.kronos_means, arts.kronos_stds)
        else:
            posterior = alpha.copy()

        confidence, path = _horizon_averaged_score(posterior, arts.transmat, arts.attractiveness, self.config.horizon)
        spot = float(posterior @ arts.attractiveness)

        return RegimeSignal(
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            ltf_belief_raw=alpha,
            ltf_belief_post_kronos=posterior,
            expected_attractiveness_path=path,
            instantaneous_attractiveness=float(np.clip(spot, 0.0, 1.0)),
            current_most_likely_ltf_state=int(np.argmax(posterior)),
        )

    def run_batch(
        self,
        df: pd.DataFrame,
        *,
        kronos_col: str | None = "kronos_forecast",
    ) -> pd.DataFrame:
        """Run inference over an entire DataFrame and append PRE columns.

        Useful for walk-forward backtesting.  The HMM forward filter is applied
        once across the whole sequence for efficiency.

        New columns added
        -----------------
        ``pre_confidence``, ``pre_ltf_state``, ``pre_spot_attractiveness``,
        ``pre_ltf_probs``, ``pre_ltf_probs_post_kronos``,
        ``pre_attractiveness_path``.
        """
        if self._artefacts is None:
            raise RuntimeError("ProbabilisticRegimeEngine must be fitted before batch scoring.")
        arts = self._artefacts
        out = df.copy()

        filtered_probs = arts.hmm.predict_proba_filtered(df)
        kronos_series: pd.Series | None = None
        if self.config.kronos_enabled and kronos_col and kronos_col in df.columns:
            kronos_series = pd.to_numeric(df[kronos_col], errors="coerce")

        confidences: list[float] = []
        spot_attrs: list[float] = []
        ltf_states: list[int] = []
        ltf_probs_raw: list[list[float]] = []
        ltf_probs_post: list[list[float]] = []
        paths: list[list[float]] = []

        for i in range(len(df)):
            kf = float(kronos_series.iloc[i]) if kronos_series is not None and np.isfinite(kronos_series.iloc[i]) else None
            sig = self.score(filtered_probs[i], kronos_forecast=kf)
            confidences.append(sig.confidence)
            spot_attrs.append(sig.instantaneous_attractiveness)
            ltf_states.append(sig.current_most_likely_ltf_state)
            ltf_probs_raw.append(sig.ltf_belief_raw.tolist())
            ltf_probs_post.append(sig.ltf_belief_post_kronos.tolist())
            paths.append(sig.expected_attractiveness_path.tolist())

        out["pre_confidence"] = confidences
        out["pre_spot_attractiveness"] = spot_attrs
        out["pre_ltf_state"] = ltf_states
        out["pre_ltf_probs"] = ltf_probs_raw
        out["pre_ltf_probs_post_kronos"] = ltf_probs_post
        out["pre_attractiveness_path"] = paths
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ProbabilisticRegimeEngine":
        return joblib.load(path)

    @property
    def is_fitted(self) -> bool:
        return self._artefacts is not None

    @property
    def attractiveness(self) -> np.ndarray | None:
        return None if self._artefacts is None else self._artefacts.attractiveness

    @property
    def transmat(self) -> np.ndarray | None:
        return None if self._artefacts is None else self._artefacts.transmat


# ---------------------------------------------------------------------------
# Multi-timeframe Probabilistic Regime Engine
# ---------------------------------------------------------------------------


class ProbabilisticRegimeEngineMTF:
    """Full multi-timeframe PRE: weekly HTF layer + daily LTF layer.

    Architecture
    ------------
    Weekly HTF filter → β_w (K_H,)
        Updated once per trading week from weekly-aggregated features.

    Daily LTF filter → α_d (K_L,)
        Updated daily; uses a time-varying mixture transition matrix:
            Ã_d = Σ_h β_w(h) * A^(L|h)
        where A^(L|h) is derived from the shared LTF transition matrix
        scaled by HTF-state-conditional persistence factors.

    Joint belief
        ψ(h,l) = β_w(h) * α_d(l)  (product assumption within week)

    Effective attractiveness
        g_eff(l) = Σ_h β_w(h) * g_LH(h, l)
        where g_LH(h, l) is attractiveness conditioned on both HTF state h
        and LTF state l.  Falls back to unconditional g_L(l) if data is sparse.

    Confidence
        C_t = (1/H) Σ_{k=0}^{H-1} (α_k)^T @ g_eff

    where α_k = α_0 @ Ã^k at each step, with a week-boundary HTF transition
    applied whenever the k-step horizon crosses a week end.

    Walkforward usage
    -----------------
    >>> mtf = ProbabilisticRegimeEngineMTF(config)
    >>> mtf.fit(ltf_df_train, htf_df_train)          # causal, on IS window
    >>> for row in live_data:
    ...     sig = mtf.update(
    ...         ltf_features=row.factor_values,
    ...         kronos_forecast=row.kronos,
    ...         is_week_boundary=row.is_week_end,
    ...         new_htf_features=row.weekly_factors if row.is_week_end else None,
    ...     )
    ...     size = sig.confidence * kelly * capital
    """

    def __init__(self, config: PREConfig | None = None) -> None:
        self.config = config or PREConfig()
        self._artefacts: _PREMTFArtefacts | None = None

        # Live streaming state (updated across update() calls)
        self._htf_belief: np.ndarray | None = None  # β_w (K_H,)
        self._ltf_belief: np.ndarray | None = None  # α_d (K_L,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame | None = None,
        *,
        kronos_col: str | None = "kronos_forecast",
        train_timestamp: str = "",
    ) -> "ProbabilisticRegimeEngineMTF":
        """Fit both the HTF and LTF regime models causally on training data.

        Parameters
        ----------
        ltf_df:
            Low-timeframe (e.g. daily) feature DataFrame.
        htf_df:
            High-timeframe (e.g. weekly) feature DataFrame.  When ``None``,
            the HTF model is built by resampling ``ltf_df`` using
            ``config.htf_resample_rule``.
        kronos_col:
            Optional Kronos column in ``ltf_df`` for conditional fitting.
        train_timestamp:
            ISO string for the training cut-off, used for artefact versioning.
        """
        cfg = self.config

        # --- HTF model ---------------------------------------------------
        if htf_df is None or htf_df.empty:
            htf_df = self._build_htf_df(ltf_df)
        htf_engine = ProbabilisticRegimeEngine(
            PREConfig(
                k_l=cfg.k_h,
                horizon=1,  # HTF scores are applied as a prior, not averaged
                r_min=cfg.r_min,
                kronos_enabled=False,
                returns_col=cfg.returns_col,
                min_state_samples=cfg.min_state_samples,
                hmm_config=cfg.htf_hmm_config,
            )
        )
        htf_engine.fit(htf_df, kronos_col=None, train_timestamp=train_timestamp)
        htf_arts = htf_engine._artefacts  # type: ignore[assignment]

        # --- LTF model ---------------------------------------------------
        ltf_engine = ProbabilisticRegimeEngine(
            PREConfig(
                k_l=cfg.k_l,
                horizon=cfg.horizon,
                r_min=cfg.r_min,
                kronos_enabled=cfg.kronos_enabled,
                returns_col=cfg.returns_col,
                min_state_samples=cfg.min_state_samples,
                kronos_min_state_samples=cfg.kronos_min_state_samples,
                hmm_config=cfg.hmm_config,
            )
        )
        ltf_engine.fit(ltf_df, kronos_col=kronos_col, train_timestamp=train_timestamp)
        ltf_arts = ltf_engine._artefacts  # type: ignore[assignment]

        self._artefacts = _PREMTFArtefacts(
            htf=htf_arts,
            ltf=ltf_arts,
            model_timestamp=train_timestamp,
        )
        # Initialise streaming beliefs to stationary distributions
        self._reset_beliefs()
        return self

    def _build_htf_df(self, ltf_df: pd.DataFrame) -> pd.DataFrame:
        """Resample LTF DataFrame to HTF by taking the last observation per period."""
        if not isinstance(ltf_df.index, pd.DatetimeIndex):
            # No datetime index: use every-N-th row as a proxy
            n = max(5, len(ltf_df) // 52)
            return ltf_df.iloc[::n].copy()
        rule = self.config.htf_resample_rule
        htf = ltf_df.resample(rule).last().dropna(how="all")
        if htf.empty or len(htf) < 4:
            # Fall back to monthly if weekly produces too few rows
            htf = ltf_df.resample("ME").last().dropna(how="all")
        return htf

    def _reset_beliefs(self) -> None:
        """Initialise streaming beliefs to uniform distributions."""
        if self._artefacts is None:
            return
        k_h = self._artefacts.htf.hmm.config.n_states
        k_l = self._artefacts.ltf.hmm.config.n_states
        self._htf_belief = np.full(k_h, 1.0 / k_h, dtype=np.float64)
        self._ltf_belief = np.full(k_l, 1.0 / k_l, dtype=np.float64)

    # ------------------------------------------------------------------
    # Inference — streaming (bar-by-bar)
    # ------------------------------------------------------------------

    def update(
        self,
        ltf_features: np.ndarray,
        kronos_forecast: float | None = None,
        *,
        is_week_boundary: bool = False,
        new_htf_features: np.ndarray | None = None,
    ) -> RegimeSignal:
        """Process one LTF bar and return an updated RegimeSignal.

        Parameters
        ----------
        ltf_features:
            Current daily factor vector (d_L,).  Must match the feature
            order used during ``fit()``.
        kronos_forecast:
            Optional scalar Kronos forecast for Bayesian sensor fusion.
        is_week_boundary:
            True if a new HTF bar has just completed (e.g. first bar of week).
        new_htf_features:
            Weekly feature vector (d_H,) when ``is_week_boundary`` is True.
            When ``None`` at a week boundary, the HTF belief is propagated
            forward via the HTF transition matrix without a new observation.
        """
        if self._artefacts is None:
            raise RuntimeError("ProbabilisticRegimeEngineMTF must be fitted before update().")
        arts = self._artefacts

        # --- 1. HTF belief update (at week boundaries) -------------------
        if is_week_boundary:
            self._htf_belief = self._update_htf_belief(new_htf_features, arts.htf)
        beta = self._htf_belief.copy()  # type: ignore[union-attr]

        # --- 2. LTF belief update ----------------------------------------
        # Mixture transition matrix: Ã = Σ_h β(h) * A^(L|h)
        # Simplification: A^(L|h) scales diagonal by HTF attractiveness
        mixture_transmat = self._mixture_transmat(beta, arts.htf.attractiveness, arts.ltf.transmat)
        new_ltf = self._step_ltf_belief(
            ltf_features=ltf_features,
            prev_belief=self._ltf_belief,  # type: ignore[arg-type]
            transmat=mixture_transmat,
            hmm=arts.ltf.hmm,
        )
        self._ltf_belief = new_ltf
        alpha = new_ltf.copy()

        # --- 3. Kronos Bayesian update ------------------------------------
        if self.config.kronos_enabled and kronos_forecast is not None and np.isfinite(kronos_forecast):
            posterior = _bayesian_kronos_update(
                alpha, kronos_forecast, arts.ltf.kronos_means, arts.ltf.kronos_stds
            )
        else:
            posterior = alpha.copy()

        # --- 4. Effective attractiveness (HTF-conditioned) ----------------
        g_eff = self._effective_attractiveness(beta, arts.htf.attractiveness, arts.ltf.attractiveness)

        # --- 5. Horizon-averaged confidence score -------------------------
        confidence, path = _horizon_averaged_score(posterior, mixture_transmat, g_eff, self.config.horizon)
        spot = float(posterior @ g_eff)

        joint = np.outer(beta, posterior).ravel()

        return RegimeSignal(
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            ltf_belief_raw=alpha,
            ltf_belief_post_kronos=posterior,
            expected_attractiveness_path=path,
            instantaneous_attractiveness=float(np.clip(spot, 0.0, 1.0)),
            current_most_likely_ltf_state=int(np.argmax(posterior)),
            htf_belief=beta,
            joint_belief=joint,
            current_most_likely_htf_state=int(np.argmax(beta)),
        )

    def _update_htf_belief(
        self,
        new_htf_features: np.ndarray | None,
        htf_arts: _PRESingleTFArtefacts,
    ) -> np.ndarray:
        """Forward-filter step for the HTF model at a week boundary."""
        prev = self._htf_belief.copy()  # type: ignore[union-attr]
        # Predictive step: propagate through HTF transition matrix
        predicted = prev @ htf_arts.transmat
        if new_htf_features is None:
            return predicted / predicted.sum()

        # Observation update: compute emission likelihoods from HTF HMM
        try:
            feature_row = pd.DataFrame([new_htf_features], columns=htf_arts.hmm.model.means_.shape and
                                       _infer_htf_columns(new_htf_features, htf_arts.hmm))
            log_ll = htf_arts.hmm.model._compute_log_likelihood(
                htf_arts.hmm.prepare_observations(feature_row)
            )
            likelihoods = np.exp(log_ll[0] - log_ll[0].max())
        except Exception:
            likelihoods = np.ones(len(predicted), dtype=np.float64)

        # Apply state permutation to likelihoods
        perm = htf_arts.hmm._state_permutation
        if perm is not None:
            ordered = np.zeros_like(likelihoods)
            for old_i, new_i in perm.items():
                if old_i < len(likelihoods) and new_i < len(ordered):
                    ordered[new_i] = likelihoods[old_i]
            likelihoods = ordered

        posterior = predicted * likelihoods
        total = posterior.sum()
        return posterior / total if total > 1e-300 else predicted

    def _step_ltf_belief(
        self,
        ltf_features: np.ndarray,
        prev_belief: np.ndarray,
        transmat: np.ndarray,
        hmm: RLMHMM,
    ) -> np.ndarray:
        """One forward-filter step for the LTF model using the mixture transmat."""
        predicted = prev_belief @ transmat
        # Compute emission likelihood from the LTF HMM
        try:
            feature_row = pd.DataFrame(
                [ltf_features],
                columns=_infer_ltf_columns(ltf_features, hmm),
            )
            log_ll = hmm.model._compute_log_likelihood(hmm.prepare_observations(feature_row))
            likelihoods = np.exp(log_ll[0] - log_ll[0].max())
        except Exception:
            likelihoods = np.ones(len(predicted), dtype=np.float64)

        # Apply state permutation
        perm = hmm._state_permutation
        if perm is not None:
            ordered = np.zeros_like(likelihoods)
            for old_i, new_i in perm.items():
                if old_i < len(likelihoods) and new_i < len(ordered):
                    ordered[new_i] = likelihoods[old_i]
            likelihoods = ordered

        posterior = predicted * likelihoods
        total = posterior.sum()
        return posterior / total if total > 1e-300 else predicted

    @staticmethod
    def _mixture_transmat(
        htf_belief: np.ndarray,
        htf_attractiveness: np.ndarray,
        ltf_transmat: np.ndarray,
    ) -> np.ndarray:
        """Derive the HTF-conditioned LTF mixture transition matrix.

        The mixture transition scales the diagonal (self-persistence) of each
        LTF state by the macro attractiveness score, then renormalises.
        This implements the concept of "LTF states are more persistent when
        the HTF environment is favourable for them."

        Effective macro attractiveness: g_H = β · g_H_vector (scalar in [0,1]).
        Diagonal scaling: T_eff[i,i] = T[i,i] * (1 + (g_H - 0.5))
        clamped so rows still sum to 1.
        """
        g_H = float(np.clip(htf_belief @ htf_attractiveness, 0.0, 1.0))
        scale = 1.0 + (g_H - 0.5)  # in [0.5, 1.5]
        T = ltf_transmat.copy()
        n = T.shape[0]
        for i in range(n):
            new_diag = T[i, i] * scale
            excess = new_diag - T[i, i]
            off_diag_total = 1.0 - T[i, i]
            if off_diag_total > 1e-12:
                T[i] -= (excess / off_diag_total) * (1.0 - np.eye(n)[i]) * T[i]
            T[i, i] = new_diag
            T[i] = np.clip(T[i], 1e-12, None)
            row_sum = T[i].sum()
            if row_sum > 1e-12:
                T[i] /= row_sum
        return T

    @staticmethod
    def _effective_attractiveness(
        htf_belief: np.ndarray,
        htf_attractiveness: np.ndarray,
        ltf_attractiveness: np.ndarray,
    ) -> np.ndarray:
        """Compute HTF-conditioned effective LTF attractiveness.

        g_eff(l) = g_L(l) * g_H_scalar
        where g_H_scalar = β_H · g_H (macro attractiveness, scalar).
        Clipped to [0, 1].
        """
        g_H = float(np.clip(htf_belief @ htf_attractiveness, 0.0, 1.0))
        return np.clip(ltf_attractiveness * g_H, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Batch inference (for walk-forward)
    # ------------------------------------------------------------------

    def run_batch(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame | None = None,
        *,
        kronos_col: str | None = "kronos_forecast",
    ) -> pd.DataFrame:
        """Apply MTF scoring across a full DataFrame (walk-forward / backtest use).

        The HTF belief is updated whenever a new HTF period completes (based on
        resampling the LTF index).  The LTF belief is updated bar-by-bar.

        New columns added
        -----------------
        ``pre_confidence``, ``pre_ltf_state``, ``pre_htf_state``,
        ``pre_spot_attractiveness``, ``pre_ltf_probs``,
        ``pre_ltf_probs_post_kronos``, ``pre_htf_probs``,
        ``pre_attractiveness_path``.
        """
        if self._artefacts is None:
            raise RuntimeError("ProbabilisticRegimeEngineMTF must be fitted before run_batch().")
        self._reset_beliefs()
        cfg = self.config
        arts = self._artefacts

        # Pre-compute HTF features aligned to LTF index
        if htf_df is None or htf_df.empty:
            htf_df = self._build_htf_df(ltf_df)
        htf_is_datetime = isinstance(ltf_df.index, pd.DatetimeIndex)

        # Map each LTF bar to its HTF period boundary flag
        is_week_end_flags = _compute_week_boundary_flags(ltf_df, cfg.htf_resample_rule)
        # Map each HTF period to its features (for updating HTF belief)
        htf_features_by_period = _build_htf_feature_lookup(htf_df)

        # Pre-compute LTF filtered probabilities for the whole batch
        ltf_filtered = arts.ltf.hmm.predict_proba_filtered(ltf_df)

        kronos_series: pd.Series | None = None
        if cfg.kronos_enabled and kronos_col and kronos_col in ltf_df.columns:
            kronos_series = pd.to_numeric(ltf_df[kronos_col], errors="coerce")

        confidences: list[float] = []
        spot_attrs: list[float] = []
        ltf_states: list[int] = []
        htf_states: list[int] = []
        ltf_probs_raw: list[list[float]] = []
        ltf_probs_post: list[list[float]] = []
        htf_probs_list: list[list[float]] = []
        paths: list[list[float]] = []

        for i in range(len(ltf_df)):
            is_wb = bool(is_week_end_flags[i])
            htf_feats: np.ndarray | None = None
            if is_wb and htf_is_datetime:
                htf_feats = _lookup_htf_features(ltf_df.index[i], htf_features_by_period, htf_df)

            kf = (
                float(kronos_series.iloc[i])
                if kronos_series is not None and np.isfinite(kronos_series.iloc[i])
                else None
            )

            # Override the HMM emission step with pre-computed filtered probs
            # (avoids redundant forward-filter computation mid-batch)
            if is_wb:
                beta_new = self._update_htf_belief(htf_feats, arts.htf)
                self._htf_belief = beta_new

            beta = self._htf_belief.copy()  # type: ignore[union-attr]
            alpha_raw = np.clip(ltf_filtered[i], 1e-12, None)
            alpha_raw /= alpha_raw.sum()
            # Update streaming LTF belief via mixture transmat then observation
            mixture_T = self._mixture_transmat(beta, arts.htf.attractiveness, arts.ltf.transmat)
            # Update stored ltf_belief (use pre-computed filtered as observation)
            predicted = self._ltf_belief @ mixture_T  # type: ignore[operator]
            # Blend prediction with the forward-filter observation
            new_ltf = 0.5 * predicted + 0.5 * alpha_raw
            new_ltf = np.clip(new_ltf, 1e-12, None)
            new_ltf /= new_ltf.sum()
            self._ltf_belief = new_ltf
            alpha = new_ltf.copy()

            # Kronos update
            if cfg.kronos_enabled and kf is not None:
                posterior = _bayesian_kronos_update(alpha, kf, arts.ltf.kronos_means, arts.ltf.kronos_stds)
            else:
                posterior = alpha.copy()

            g_eff = self._effective_attractiveness(beta, arts.htf.attractiveness, arts.ltf.attractiveness)
            confidence, path = _horizon_averaged_score(posterior, mixture_T, g_eff, cfg.horizon)

            confidences.append(float(np.clip(confidence, 0.0, 1.0)))
            spot_attrs.append(float(np.clip(posterior @ g_eff, 0.0, 1.0)))
            ltf_states.append(int(np.argmax(posterior)))
            htf_states.append(int(np.argmax(beta)))
            ltf_probs_raw.append(alpha.tolist())
            ltf_probs_post.append(posterior.tolist())
            htf_probs_list.append(beta.tolist())
            paths.append(path.tolist())

        out = ltf_df.copy()
        out["pre_confidence"] = confidences
        out["pre_spot_attractiveness"] = spot_attrs
        out["pre_ltf_state"] = ltf_states
        out["pre_htf_state"] = htf_states
        out["pre_ltf_probs"] = ltf_probs_raw
        out["pre_ltf_probs_post_kronos"] = ltf_probs_post
        out["pre_htf_probs"] = htf_probs_list
        out["pre_attractiveness_path"] = paths
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ProbabilisticRegimeEngineMTF":
        return joblib.load(path)

    @property
    def is_fitted(self) -> bool:
        return self._artefacts is not None

    @property
    def attractiveness(self) -> np.ndarray | None:
        return None if self._artefacts is None else self._artefacts.ltf.attractiveness

    @property
    def htf_attractiveness(self) -> np.ndarray | None:
        return None if self._artefacts is None else self._artefacts.htf.attractiveness


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _infer_htf_columns(features: np.ndarray, hmm: RLMHMM) -> list[str]:
    """Infer column names for a raw feature vector, defaulting to S_D/S_V/S_L/S_G."""
    required = ["S_D", "S_V", "S_L", "S_G"]
    if len(features) == 4:
        return required
    # Pad/truncate to standard 4 columns for HMM compatibility
    return required[: len(features)] + [f"_f{i}" for i in range(max(0, len(features) - 4))]


def _infer_ltf_columns(features: np.ndarray, hmm: RLMHMM) -> list[str]:
    return _infer_htf_columns(features, hmm)


def _compute_week_boundary_flags(df: pd.DataFrame, rule: str) -> np.ndarray:
    """Return boolean array: True on the first LTF bar of each HTF period."""
    flags = np.zeros(len(df), dtype=bool)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Non-datetime: every 5th bar is a "week boundary"
        period = 5
        for i in range(0, len(df), period):
            flags[i] = True
        return flags
    try:
        periods = df.index.to_period(rule.split("-")[0])
        prev = None
        for i, p in enumerate(periods):
            if p != prev:
                flags[i] = True
                prev = p
    except Exception:
        flags[0] = True
    return flags


def _build_htf_feature_lookup(htf_df: pd.DataFrame) -> dict:
    """Build a simple dict mapping HTF index → numeric feature array for lookup."""
    if htf_df.empty:
        return {}
    numeric_df = htf_df.select_dtypes(include=[np.number])
    return {idx: row.values.astype(np.float64) for idx, row in numeric_df.iterrows()}


def _lookup_htf_features(
    ltf_timestamp: object,
    lookup: dict,
    htf_df: pd.DataFrame,
) -> np.ndarray | None:
    """Find the most recent HTF row at or before the given LTF timestamp."""
    if not lookup:
        return None
    if isinstance(ltf_timestamp, pd.Timestamp):
        candidates = [k for k in lookup if isinstance(k, pd.Timestamp) and k <= ltf_timestamp]
        if not candidates:
            return None
        latest = max(candidates)
        return lookup[latest]
    return None


# ---------------------------------------------------------------------------
# Decision-layer helper: extract PRE confidence from a data row
# ---------------------------------------------------------------------------


def extract_pre_confidence(row: "pd.Series") -> float | None:  # noqa: F821
    """Return ``pre_confidence`` from a DataFrame row, or ``None`` if absent.

    This is the integration point for ``compute_regime_modulators`` in
    ``rlm.roee.decision``.  When a non-null ``pre_confidence`` is present, the
    decision layer uses it directly instead of computing the binary gate.
    """
    val = row.get("pre_confidence", None)
    if val is None:
        return None
    try:
        import math

        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
