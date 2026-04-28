from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def svi_raw(
    k: np.ndarray | float, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    x = np.asarray(k, dtype=float)
    return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma**2))


def _svi_first_derivative(k: float, *, b: float, rho: float, m: float, sigma: float) -> float:
    diff = float(k - m)
    denom = np.sqrt(diff**2 + sigma**2)
    return float(b * (rho + (diff / denom)))


def _svi_second_derivative(k: float, *, b: float, m: float, sigma: float) -> float:
    diff = float(k - m)
    denom = (diff**2 + sigma**2) ** 1.5
    if denom <= 1e-12:
        return 0.0
    return float(b * (sigma**2) / denom)


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    tau_years: float
    dte: float
    rmse: float

    def total_variance(self, k: float) -> float:
        return float(svi_raw(np.array([k]), self.a, self.b, self.rho, self.m, self.sigma)[0])

    def atm_iv(self) -> float:
        if self.tau_years <= 1e-9:
            return float("nan")
        return float(np.sqrt(max(self.total_variance(0.0), 0.0) / self.tau_years))

    def skew(self) -> float:
        if self.tau_years <= 1e-9:
            return 0.0
        return (
            _svi_first_derivative(
                0.0,
                b=self.b,
                rho=self.rho,
                m=self.m,
                sigma=self.sigma,
            )
            / self.tau_years
        )

    def convexity(self) -> float:
        if self.tau_years <= 1e-9:
            return 0.0
        return (
            _svi_second_derivative(
                0.0,
                b=self.b,
                m=self.m,
                sigma=self.sigma,
            )
            / self.tau_years
        )


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = v.notna() & w.notna()
    if not mask.any():
        return float("nan")
    w_masked = w.loc[mask]
    if float(w_masked.sum()) <= 1e-9:
        return float(v.loc[mask].mean())
    return float(np.average(v.loc[mask], weights=w_masked))


def fit_svi_for_expiry(
    expiry_slice: pd.DataFrame,
    *,
    spot: float,
    min_points: int = 5,
) -> SVIParams | None:
    if (
        expiry_slice.empty
        or not np.isfinite(spot)
        or spot <= 0.0
        or "iv" not in expiry_slice.columns
    ):
        return None

    tau_years = float(pd.to_numeric(expiry_slice["dte"], errors="coerce").median()) / 365.0
    if not np.isfinite(tau_years) or tau_years <= 0.0:
        return None

    work = expiry_slice.copy()
    if "open_interest" in work.columns:
        weights = pd.to_numeric(work["open_interest"], errors="coerce").fillna(0.0).clip(lower=1.0)
    elif "volume" in work.columns:
        weights = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0).clip(lower=1.0)
    else:
        weights = pd.Series(1.0, index=work.index)
    work["_weight"] = weights

    rows: list[dict[str, float]] = []
    for strike, grp in work.groupby("strike", sort=True):
        iv = _weighted_mean(grp["iv"], grp["_weight"])
        if not np.isfinite(iv) or iv <= 0.0:
            continue
        rows.append(
            {
                "strike": float(strike),
                "log_moneyness": float(np.log(float(strike) / float(spot))),
                "total_variance": float(iv**2 * tau_years),
            }
        )

    if len(rows) < min_points:
        return None

    sample = pd.DataFrame(rows).sort_values("log_moneyness")
    k = sample["log_moneyness"].to_numpy(dtype=float)
    w = sample["total_variance"].to_numpy(dtype=float)

    p0 = (
        float(np.nanmedian(w)),
        0.15,
        -0.3,
        0.0,
        0.15,
    )
    bounds = (
        [1e-8, 1e-6, -0.999, -2.0, 1e-4],
        [5.0, 5.0, 0.999, 2.0, 2.0],
    )

    try:
        params, _ = curve_fit(
            svi_raw,
            k,
            w,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )
    except Exception:
        return None

    fitted = svi_raw(k, *params)
    rmse = float(np.sqrt(np.mean((w - fitted) ** 2)))
    return SVIParams(
        a=float(params[0]),
        b=float(params[1]),
        rho=float(params[2]),
        m=float(params[3]),
        sigma=float(params[4]),
        tau_years=tau_years,
        dte=float(pd.to_numeric(expiry_slice["dte"], errors="coerce").median()),
        rmse=rmse,
    )


def extract_surface_features(
    chain_slice: pd.DataFrame,
    *,
    spot: float,
    target_dte: float = 30.0,
) -> dict[str, float]:
    neutral = {
        "surface_atm_iv": float("nan"),
        "surface_variance": float("nan"),
        "surface_skew": float("nan"),
        "surface_convexity": float("nan"),
        "surface_term_structure_ratio": float("nan"),
        "surface_fit_error": float("nan"),
        "surface_selected_dte": float("nan"),
    }
    if chain_slice.empty or "iv" not in chain_slice.columns:
        return neutral

    fits: list[SVIParams] = []
    for _, grp in chain_slice.groupby("expiry", sort=True):
        params = fit_svi_for_expiry(grp, spot=spot)
        if params is not None:
            fits.append(params)

    if not fits:
        return neutral

    selected = min(fits, key=lambda x: (abs(x.dte - target_dte), x.dte))
    short_ivs = [
        fit.atm_iv() for fit in fits if 14.0 <= fit.dte <= 45.0 and np.isfinite(fit.atm_iv())
    ]
    long_ivs = [
        fit.atm_iv() for fit in fits if 46.0 <= fit.dte <= 120.0 and np.isfinite(fit.atm_iv())
    ]
    if short_ivs and long_ivs and float(np.nanmedian(long_ivs)) > 1e-9:
        term_structure = float(np.nanmedian(short_ivs) / np.nanmedian(long_ivs))
    else:
        term_structure = float("nan")

    atm_iv = selected.atm_iv()
    return {
        "surface_atm_iv": atm_iv,
        "surface_variance": float(atm_iv**2) if np.isfinite(atm_iv) else float("nan"),
        "surface_skew": selected.skew(),
        "surface_convexity": selected.convexity(),
        "surface_term_structure_ratio": term_structure,
        "surface_fit_error": float(selected.rmse),
        "surface_selected_dte": float(selected.dte),
    }


def build_surface_feature_frame(
    chain: pd.DataFrame,
    *,
    target_dte: float = 30.0,
) -> pd.DataFrame:
    """
    Build one row of SVI-derived features per calendar date.
    """
    if chain.empty:
        return pd.DataFrame()

    required = {"timestamp", "strike", "expiry", "dte", "iv"}
    missing = required.difference(chain.columns)
    if missing:
        raise ValueError(f"Missing required chain columns for surface features: {sorted(missing)}")

    work = chain.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work["_d"] = work["timestamp"].dt.normalize()

    if "underlying_price" in work.columns:
        spot_by_day = (
            pd.to_numeric(work["underlying_price"], errors="coerce").groupby(work["_d"]).median()
        )
    else:
        spot_by_day = pd.Series(dtype=float)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for day, grp in work.groupby("_d", sort=True):
        if day in spot_by_day.index and np.isfinite(float(spot_by_day.loc[day])):
            spot = float(spot_by_day.loc[day])
        else:
            if grp.empty:
                continue
            spot = float(pd.to_numeric(grp["strike"], errors="coerce").median())
        if not np.isfinite(spot) or spot <= 0.0:
            continue

        feats = extract_surface_features(grp, spot=spot, target_dte=target_dte)
        rows.append(
            {
                "_d": pd.Timestamp(day),
                "surface_atm_forward_iv": feats["surface_atm_iv"],
                "surface_skew": feats["surface_skew"],
                "surface_convexity": feats["surface_convexity"],
                "surface_term_slope": feats["surface_term_structure_ratio"],
                "surface_fit_error": feats["surface_fit_error"],
                "surface_selected_dte": feats["surface_selected_dte"],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("_d").sort_index()
