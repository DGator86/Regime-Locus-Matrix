from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rlm.features.scoring.coordinate_mapper import add_market_coordinate_columns
from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.types.coordinates import MarketCoordinate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COORD_OUT_COLS = [
    "M_D", "M_V", "M_L", "M_G",
    "M_trend_strength", "M_dealer_control", "M_alignment", "M_delta_neutral",
    "M_lat_D", "M_lat_V", "M_lat_L", "M_lat_G",
    "M_R_trans",
]


def _score_df(**kwargs: list) -> pd.DataFrame:
    base = {
        "S_D": [0.0, 0.0],
        "S_V": [0.0, 0.0],
        "S_L": [0.0, 0.0],
        "S_G": [0.0, 0.0],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# MarketCoordinate.from_scores
# ---------------------------------------------------------------------------


def test_from_scores_neutral() -> None:
    m = MarketCoordinate.from_scores(0.0, 0.0, 0.0, 0.0)
    assert m is not None
    assert m == MarketCoordinate.neutral()


def test_from_scores_extremes() -> None:
    m = MarketCoordinate.from_scores(1.0, 1.0, 1.0, 1.0)
    assert m is not None
    assert m.D == pytest.approx(10.0)
    assert m.V == pytest.approx(10.0)
    assert m.L == pytest.approx(10.0)
    assert m.G == pytest.approx(10.0)


def test_from_scores_negative_extremes() -> None:
    m = MarketCoordinate.from_scores(-1.0, -1.0, -1.0, -1.0)
    assert m is not None
    assert m.D == pytest.approx(0.0)


def test_from_scores_clamps_beyond_unit_interval() -> None:
    # scores outside [-1, 1] (theoretical, can happen if a factor saturates)
    m = MarketCoordinate.from_scores(2.0, -3.0, 0.0, 0.0)
    assert m is not None
    assert m.D == pytest.approx(10.0)
    assert m.V == pytest.approx(0.0)


def test_from_scores_partial_values() -> None:
    m = MarketCoordinate.from_scores(0.6, -0.4, 0.0, 0.8)
    assert m is not None
    assert m.D == pytest.approx(8.0)
    assert m.V == pytest.approx(3.0)
    assert m.L == pytest.approx(5.0)
    assert m.G == pytest.approx(9.0)


def test_from_scores_nan_returns_none() -> None:
    assert MarketCoordinate.from_scores(float("nan"), 0.0, 0.0, 0.0) is None
    assert MarketCoordinate.from_scores(0.0, float("nan"), 0.0, 0.0) is None


def test_from_scores_inf_returns_none() -> None:
    assert MarketCoordinate.from_scores(float("inf"), 0.0, 0.0, 0.0) is None
    assert MarketCoordinate.from_scores(0.0, 0.0, float("-inf"), 0.0) is None


# ---------------------------------------------------------------------------
# add_market_coordinate_columns — column presence
# ---------------------------------------------------------------------------


def test_all_output_columns_present() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    for col in _COORD_OUT_COLS:
        assert col in out.columns, f"Missing column: {col}"


def test_missing_score_columns_raises() -> None:
    df = pd.DataFrame({"S_D": [0.0], "S_V": [0.0]})  # missing S_L, S_G
    with pytest.raises(ValueError, match="missing score columns"):
        add_market_coordinate_columns(df)


def test_input_columns_preserved() -> None:
    df = pd.DataFrame({"S_D": [0.0], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0], "close": [100.0]})
    out = add_market_coordinate_columns(df)
    assert "close" in out.columns


# ---------------------------------------------------------------------------
# add_market_coordinate_columns — coordinate axis values
# ---------------------------------------------------------------------------


def test_coord_neutral_scores_give_five() -> None:
    df = _score_df(S_D=[0.0], S_V=[0.0], S_L=[0.0], S_G=[0.0])
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(5.0)
    assert out["M_V"].iloc[0] == pytest.approx(5.0)
    assert out["M_L"].iloc[0] == pytest.approx(5.0)
    assert out["M_G"].iloc[0] == pytest.approx(5.0)


def test_coord_max_score_gives_ten() -> None:
    df = pd.DataFrame({"S_D": [1.0], "S_V": [1.0], "S_L": [1.0], "S_G": [1.0]})
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(10.0)


def test_coord_min_score_gives_zero() -> None:
    df = pd.DataFrame({"S_D": [-1.0], "S_V": [-1.0], "S_L": [-1.0], "S_G": [-1.0]})
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(0.0)


def test_coord_linear_mapping() -> None:
    # S_D = 0.6 → M_D = 5 + 5*0.6 = 8.0
    df = pd.DataFrame({"S_D": [0.6], "S_V": [-0.4], "S_L": [0.0], "S_G": [0.8]})
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(8.0)
    assert out["M_V"].iloc[0] == pytest.approx(3.0)
    assert out["M_L"].iloc[0] == pytest.approx(5.0)
    assert out["M_G"].iloc[0] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# add_market_coordinate_columns — derived invariants
# ---------------------------------------------------------------------------


def test_trend_strength_zero_at_neutral() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    assert out["M_trend_strength"].iloc[0] == pytest.approx(0.0)


def test_trend_strength_bullish() -> None:
    df = pd.DataFrame({"S_D": [0.6], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0]})
    out = add_market_coordinate_columns(df)
    # M_D = 8.0 → trend_strength = 3.0
    assert out["M_trend_strength"].iloc[0] == pytest.approx(3.0)


def test_alignment_positive_reinforcement() -> None:
    # Bull (S_D=0.6 → d=3) + Stabilising (S_G=0.6 → g=3) → A = 9
    df = pd.DataFrame({"S_D": [0.6], "S_V": [0.0], "S_L": [0.0], "S_G": [0.6]})
    out = add_market_coordinate_columns(df)
    assert out["M_alignment"].iloc[0] == pytest.approx(9.0)


def test_alignment_negative_opposition() -> None:
    # Bull (d=3) + Destabilising (g=-3) → A = -9
    df = pd.DataFrame({"S_D": [0.6], "S_V": [0.0], "S_L": [0.0], "S_G": [-0.6]})
    out = add_market_coordinate_columns(df)
    assert out["M_alignment"].iloc[0] == pytest.approx(-9.0)


def test_delta_neutral_at_origin() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    assert out["M_delta_neutral"].iloc[0] == pytest.approx(0.0)


def test_delta_neutral_all_axes_max() -> None:
    # All axes displaced by 5 → sqrt(4 * 25) = 10
    df = pd.DataFrame({"S_D": [1.0], "S_V": [1.0], "S_L": [1.0], "S_G": [1.0]})
    out = add_market_coordinate_columns(df)
    assert out["M_delta_neutral"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# add_market_coordinate_columns — lattice bins
# ---------------------------------------------------------------------------


def test_lattice_bins_neutral_is_mid() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    assert int(out["M_lat_D"].iloc[0]) == 2
    assert int(out["M_lat_V"].iloc[0]) == 2
    assert int(out["M_lat_L"].iloc[0]) == 2
    assert int(out["M_lat_G"].iloc[0]) == 2


def test_lattice_bins_low() -> None:
    # S_D = -1 → M_D = 0 → bin 1
    df = pd.DataFrame({"S_D": [-1.0], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0]})
    out = add_market_coordinate_columns(df)
    assert int(out["M_lat_D"].iloc[0]) == 1


def test_lattice_bins_high() -> None:
    # S_D = 1 → M_D = 10 → bin 3
    df = pd.DataFrame({"S_D": [1.0], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0]})
    out = add_market_coordinate_columns(df)
    assert int(out["M_lat_D"].iloc[0]) == 3


def test_lattice_bin_at_low_boundary() -> None:
    # M_D = 3.0 (exactly) → bin 1
    s = (3.0 - 5.0) / 5.0  # s = -0.4
    df = pd.DataFrame({"S_D": [s], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0]})
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(3.0)
    assert int(out["M_lat_D"].iloc[0]) == 1


def test_lattice_bin_at_high_boundary() -> None:
    # M_D = 7.0 (exactly) → bin 3
    s = (7.0 - 5.0) / 5.0  # s = 0.4
    df = pd.DataFrame({"S_D": [s], "S_V": [0.0], "S_L": [0.0], "S_G": [0.0]})
    out = add_market_coordinate_columns(df)
    assert out["M_D"].iloc[0] == pytest.approx(7.0)
    assert int(out["M_lat_D"].iloc[0]) == 3


# ---------------------------------------------------------------------------
# add_market_coordinate_columns — transition magnitude
# ---------------------------------------------------------------------------


def test_R_trans_first_row_is_nan() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    assert math.isnan(out["M_R_trans"].iloc[0])


def test_R_trans_same_state_is_zero() -> None:
    df = _score_df()
    out = add_market_coordinate_columns(df)
    assert out["M_R_trans"].iloc[1] == pytest.approx(0.0)


def test_R_trans_one_axis_change() -> None:
    # M_D moves from 5 (s=0) to 8 (s=0.6): ΔD = 3 → R_trans = 3
    df = pd.DataFrame({
        "S_D": [0.0, 0.6],
        "S_V": [0.0, 0.0],
        "S_L": [0.0, 0.0],
        "S_G": [0.0, 0.0],
    })
    out = add_market_coordinate_columns(df)
    assert out["M_R_trans"].iloc[1] == pytest.approx(3.0)


def test_R_trans_two_axis_change() -> None:
    # ΔD = 3, ΔV = 4 → R_trans = 5 (3-4-5 right triangle)
    s_d_1 = 0.6   # M_D = 8
    s_v_1 = -0.2  # M_V = 4, so ΔV from 5 to 4... let me compute
    # t=0: M_D=5, M_V=5
    # t=1: M_D=8 (s=0.6), M_V=9 (s=0.8)  → ΔD=3, ΔV=4 → R=5
    df = pd.DataFrame({
        "S_D": [0.0, 0.6],
        "S_V": [0.0, 0.8],
        "S_L": [0.0, 0.0],
        "S_G": [0.0, 0.0],
    })
    out = add_market_coordinate_columns(df)
    assert out["M_R_trans"].iloc[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# NaN propagation (warmup rows)
# ---------------------------------------------------------------------------


def test_nan_scores_produce_nan_coordinates() -> None:
    df = pd.DataFrame({
        "S_D": [float("nan")],
        "S_V": [0.0],
        "S_L": [0.0],
        "S_G": [0.0],
    })
    out = add_market_coordinate_columns(df)
    assert math.isnan(out["M_D"].iloc[0])
    assert math.isnan(out["M_delta_neutral"].iloc[0])


# ---------------------------------------------------------------------------
# classify_state_matrix — coordinate columns wired in
# ---------------------------------------------------------------------------


def test_classify_state_matrix_includes_coordinate_columns() -> None:
    df = pd.DataFrame({
        "S_D": [0.8, -0.8, 0.0],
        "S_V": [-0.5, 0.7, 0.0],
        "S_L": [0.6, -0.2, 0.0],
        "S_G": [0.7, -0.8, 0.0],
    })
    out = classify_state_matrix(df)
    for col in _COORD_OUT_COLS:
        assert col in out.columns, f"classify_state_matrix missing column: {col}"


def test_classify_state_matrix_coordinate_values_match_mapper() -> None:
    df = pd.DataFrame({
        "S_D": [0.6],
        "S_V": [0.0],
        "S_L": [0.0],
        "S_G": [-0.6],
    })
    out = classify_state_matrix(df)
    assert out["M_D"].iloc[0] == pytest.approx(8.0)
    assert out["M_G"].iloc[0] == pytest.approx(2.0)
    assert out["M_alignment"].iloc[0] == pytest.approx(-9.0)


def test_classify_state_matrix_still_produces_regime_columns() -> None:
    df = pd.DataFrame({
        "S_D": [0.8],
        "S_V": [0.5],
        "S_L": [0.6],
        "S_G": [0.7],
    })
    out = classify_state_matrix(df)
    assert out["direction_regime"].iloc[0] == "bull"
    assert "regime_key" in out.columns


# ---------------------------------------------------------------------------
# from_scores round-trip: MarketCoordinate ↔ add_market_coordinate_columns
# ---------------------------------------------------------------------------


def test_from_scores_matches_dataframe_mapper() -> None:
    s_d, s_v, s_l, s_g = 0.4, -0.3, 0.6, -0.1
    m = MarketCoordinate.from_scores(s_d, s_v, s_l, s_g)
    assert m is not None

    df = pd.DataFrame({"S_D": [s_d], "S_V": [s_v], "S_L": [s_l], "S_G": [s_g]})
    out = add_market_coordinate_columns(df)

    assert out["M_D"].iloc[0] == pytest.approx(m.D)
    assert out["M_V"].iloc[0] == pytest.approx(m.V)
    assert out["M_L"].iloc[0] == pytest.approx(m.L)
    assert out["M_G"].iloc[0] == pytest.approx(m.G)
    assert out["M_trend_strength"].iloc[0] == pytest.approx(m.trend_strength)
    assert out["M_dealer_control"].iloc[0] == pytest.approx(m.dealer_control_magnitude)
    assert out["M_alignment"].iloc[0] == pytest.approx(m.direction_dealer_alignment)
    assert out["M_delta_neutral"].iloc[0] == pytest.approx(
        m.weighted_distance_from_neutral()
    )
    i, j, k, l = m.lattice_index()
    assert int(out["M_lat_D"].iloc[0]) == i
    assert int(out["M_lat_V"].iloc[0]) == j
    assert int(out["M_lat_L"].iloc[0]) == k
    assert int(out["M_lat_G"].iloc[0]) == l
