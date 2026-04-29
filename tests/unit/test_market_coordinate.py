from __future__ import annotations

import math

import pytest

from rlm.types.coordinates import MarketCoordinate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mc(D: float = 5.0, V: float = 5.0, L: float = 5.0, G: float = 5.0) -> MarketCoordinate:
    return MarketCoordinate(D=D, V=V, L=L, G=G)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_valid_construction() -> None:
    m = mc(D=0.0, V=10.0, L=3.5, G=7.2)
    assert m.D == 0.0
    assert m.V == 10.0
    assert m.L == 3.5
    assert m.G == 7.2


def test_boundary_values_accepted() -> None:
    mc(D=0.0, V=0.0, L=0.0, G=0.0)
    mc(D=10.0, V=10.0, L=10.0, G=10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"D": -0.1},
        {"D": 10.1},
        {"V": -1.0},
        {"V": 11.0},
        {"L": -0.001},
        {"G": 10.5},
    ],
)
def test_out_of_range_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="outside \\[0, 10\\]"):
        mc(**kwargs)


def test_immutability() -> None:
    m = mc()
    with pytest.raises((AttributeError, TypeError)):
        m.D = 3.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Neutral factory
# ---------------------------------------------------------------------------


def test_neutral_factory() -> None:
    m0 = MarketCoordinate.neutral()
    assert m0.D == 5.0
    assert m0.V == 5.0
    assert m0.L == 5.0
    assert m0.G == 5.0


# ---------------------------------------------------------------------------
# Centred coordinates
# ---------------------------------------------------------------------------


def test_d_centred() -> None:
    assert mc(D=5.0).d == pytest.approx(0.0)
    assert mc(D=0.0).d == pytest.approx(-5.0)
    assert mc(D=10.0).d == pytest.approx(5.0)
    assert mc(D=8.0).d == pytest.approx(3.0)


def test_g_centred() -> None:
    assert mc(G=5.0).g == pytest.approx(0.0)
    assert mc(G=0.0).g == pytest.approx(-5.0)
    assert mc(G=10.0).g == pytest.approx(5.0)
    assert mc(G=2.5).g == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Trend strength T = |d|
# ---------------------------------------------------------------------------


def test_trend_strength_neutral_is_zero() -> None:
    assert mc(D=5.0).trend_strength == pytest.approx(0.0)


def test_trend_strength_bullish() -> None:
    assert mc(D=8.0).trend_strength == pytest.approx(3.0)


def test_trend_strength_bearish_same_magnitude() -> None:
    assert mc(D=2.0).trend_strength == pytest.approx(3.0)


def test_trend_strength_max() -> None:
    assert mc(D=0.0).trend_strength == pytest.approx(5.0)
    assert mc(D=10.0).trend_strength == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Dealer control magnitude C = |g|
# ---------------------------------------------------------------------------


def test_dealer_control_neutral() -> None:
    assert mc(G=5.0).dealer_control_magnitude == pytest.approx(0.0)


def test_dealer_control_stabilising() -> None:
    assert mc(G=8.0).dealer_control_magnitude == pytest.approx(3.0)


def test_dealer_control_destabilising_same_magnitude() -> None:
    assert mc(G=2.0).dealer_control_magnitude == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Direction–dealer alignment A_DG = d · g
# ---------------------------------------------------------------------------


def test_alignment_both_neutral_is_zero() -> None:
    assert mc(D=5.0, G=5.0).direction_dealer_alignment == pytest.approx(0.0)


def test_alignment_positive_bull_stabilising() -> None:
    # d=3, g=3 → A=9
    assert mc(D=8.0, G=8.0).direction_dealer_alignment == pytest.approx(9.0)


def test_alignment_positive_bear_destabilising() -> None:
    # d=-3, g=-3 → A=9
    assert mc(D=2.0, G=2.0).direction_dealer_alignment == pytest.approx(9.0)


def test_alignment_negative_bull_destabilising() -> None:
    # d=3, g=-3 → A=-9
    assert mc(D=8.0, G=2.0).direction_dealer_alignment == pytest.approx(-9.0)


def test_alignment_negative_bear_stabilising() -> None:
    # d=-3, g=3 → A=-9
    assert mc(D=2.0, G=8.0).direction_dealer_alignment == pytest.approx(-9.0)


def test_alignment_bounds() -> None:
    # max = 5 * 5 = 25
    assert mc(D=10.0, G=10.0).direction_dealer_alignment == pytest.approx(25.0)
    assert mc(D=0.0, G=0.0).direction_dealer_alignment == pytest.approx(25.0)
    # min = -25
    assert mc(D=10.0, G=0.0).direction_dealer_alignment == pytest.approx(-25.0)
    assert mc(D=0.0, G=10.0).direction_dealer_alignment == pytest.approx(-25.0)


# ---------------------------------------------------------------------------
# Weighted distance from neutral
# ---------------------------------------------------------------------------


def test_distance_neutral_is_zero() -> None:
    assert MarketCoordinate.neutral().weighted_distance_from_neutral() == pytest.approx(0.0)


def test_distance_unit_weights_one_axis() -> None:
    # Only D displaced by 5
    m = mc(D=10.0)
    assert m.weighted_distance_from_neutral() == pytest.approx(5.0)


def test_distance_all_axes_equal() -> None:
    # All axes displaced by 5 → sqrt(4 * 25) = 10
    m = mc(D=10.0, V=10.0, L=10.0, G=10.0)
    assert m.weighted_distance_from_neutral() == pytest.approx(10.0)


def test_distance_custom_weights() -> None:
    # D displaced by 4, wD=4 → sqrt(4*16) = 8
    m = mc(D=9.0)
    result = m.weighted_distance_from_neutral(wD=4.0, wV=1.0, wL=1.0, wG=1.0)
    assert result == pytest.approx(math.sqrt(4 * 16))


def test_distance_zero_weight_excludes_axis() -> None:
    # D displaced by 5 but wD=0 → distance driven only by others
    m = mc(D=0.0, V=5.0, L=5.0, G=5.0)
    assert m.weighted_distance_from_neutral(wD=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Transition magnitude
# ---------------------------------------------------------------------------


def test_transition_magnitude_same_state() -> None:
    m = mc(D=3.0, V=7.0, L=4.0, G=6.0)
    assert m.transition_magnitude(m) == pytest.approx(0.0)


def test_transition_magnitude_one_axis() -> None:
    prev = mc(D=5.0)
    curr = mc(D=8.0)
    assert curr.transition_magnitude(prev) == pytest.approx(3.0)


def test_transition_magnitude_two_axes() -> None:
    prev = mc(D=5.0, V=5.0)
    curr = mc(D=8.0, V=9.0)
    # sqrt(9 + 16) = 5
    assert curr.transition_magnitude(prev) == pytest.approx(5.0)


def test_transition_magnitude_weighted() -> None:
    prev = mc(D=5.0, V=5.0)
    curr = mc(D=6.0, V=7.0)
    # wD=2, wV=2: sqrt(2*1 + 2*4) = sqrt(10)
    result = curr.transition_magnitude(prev, wD=2.0, wV=2.0, wL=2.0, wG=2.0)
    assert result == pytest.approx(math.sqrt(2 * 1 + 2 * 4))


# ---------------------------------------------------------------------------
# Lattice index — bin assignment
# ---------------------------------------------------------------------------


def test_lattice_index_neutral_is_mid() -> None:
    assert MarketCoordinate.neutral().lattice_index() == (2, 2, 2, 2)


def test_lattice_index_all_low() -> None:
    m = mc(D=0.0, V=0.0, L=0.0, G=0.0)
    assert m.lattice_index() == (1, 1, 1, 1)


def test_lattice_index_all_high() -> None:
    m = mc(D=10.0, V=10.0, L=10.0, G=10.0)
    assert m.lattice_index() == (3, 3, 3, 3)


def test_lattice_index_boundaries() -> None:
    # At exactly low_bound (3.0) → bin 1
    assert mc(D=3.0).lattice_index()[0] == 1
    # Just above low_bound → bin 2
    assert mc(D=3.01).lattice_index()[0] == 2
    # At exactly high_bound (7.0) → bin 3
    assert mc(D=7.0).lattice_index()[0] == 3
    # Just below high_bound → bin 2
    assert mc(D=6.99).lattice_index()[0] == 2


def test_lattice_index_mixed() -> None:
    m = mc(D=2.0, V=5.0, L=8.0, G=1.0)
    assert m.lattice_index() == (1, 2, 3, 1)


def test_lattice_index_custom_bounds() -> None:
    m = mc(D=4.0, V=4.0, L=4.0, G=4.0)
    # With default bounds (3, 7): 4 is in bin 2
    assert m.lattice_index(low_bound=3.0, high_bound=7.0) == (2, 2, 2, 2)
    # With tighter bounds (5, 5): 4 ≤ 5 is bin 1
    assert m.lattice_index(low_bound=5.0, high_bound=5.0)[0] == 1


# ---------------------------------------------------------------------------
# Lattice distance
# ---------------------------------------------------------------------------


def test_lattice_distance_same_cell_is_zero() -> None:
    m = mc(D=2.0, V=5.0, L=8.0, G=1.0)
    assert m.lattice_distance(m) == 0


def test_lattice_distance_adjacent() -> None:
    # D moves from low (2) to mid (5): one step
    a = mc(D=2.0)
    b = mc(D=5.0)
    assert a.lattice_distance(b) == 1


def test_lattice_distance_opposite_corners() -> None:
    a = mc(D=0.0, V=0.0, L=0.0, G=0.0)  # (1,1,1,1)
    b = mc(D=10.0, V=10.0, L=10.0, G=10.0)  # (3,3,3,3)
    assert a.lattice_distance(b) == 8


def test_lattice_distance_two_axes() -> None:
    a = mc(D=0.0, V=0.0)  # D=1, V=1
    b = mc(D=10.0, V=10.0)  # D=3, V=3
    assert a.lattice_distance(b) == 4


def test_lattice_distance_is_symmetric() -> None:
    a = mc(D=1.0, V=9.0, L=5.0, G=2.0)
    b = mc(D=8.0, V=2.0, L=5.0, G=9.0)
    assert a.lattice_distance(b) == b.lattice_distance(a)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_contains_values() -> None:
    m = mc(D=3.5, V=7.2, L=1.0, G=9.9)
    r = repr(m)
    assert "3.500" in r
    assert "7.200" in r
    assert "1.000" in r
    assert "9.900" in r
