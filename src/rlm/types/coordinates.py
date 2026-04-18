from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketCoordinate:
    """
    A single point in the 4D market state space M = [0,10]^4.

    Axes
    ----
    D : direction     — 0 = max bearish, 5 = neutral, 10 = max bullish
    V : volatility    — 0 = max compression, 5 = normal, 10 = max expansion
    L : liquidity     — 0 = illiquid / wide spreads, 10 = excellent execution
    G : dealer flow   — 0 = destabilising / move-amplifying, 5 = neutral,
                        10 = stabilising / mean-reverting

    Neutral point  M_0 = (5, 5, 5, 5).
    """

    D: float  # direction   [0, 10]
    V: float  # volatility  [0, 10]
    L: float  # liquidity   [0, 10]
    G: float  # dealer flow [0, 10]

    def __post_init__(self) -> None:
        for name, val in (("D", self.D), ("V", self.V), ("L", self.L), ("G", self.G)):
            if not (0.0 <= val <= 10.0):
                raise ValueError(f"MarketCoordinate.{name}={val!r} is outside [0, 10]")

    # ------------------------------------------------------------------
    # Centred coordinates
    # ------------------------------------------------------------------

    @property
    def d(self) -> float:
        """d = D − 5 ∈ [−5, 5].  Negative = bearish, positive = bullish."""
        return self.D - 5.0

    @property
    def g(self) -> float:
        """g = G − 5 ∈ [−5, 5].  Negative = destabilising, positive = stabilising."""
        return self.G - 5.0

    # ------------------------------------------------------------------
    # Scalar invariants
    # ------------------------------------------------------------------

    @property
    def trend_strength(self) -> float:
        """T = |d| ∈ [0, 5].  Directional conviction independent of sign."""
        return abs(self.d)

    @property
    def dealer_control_magnitude(self) -> float:
        """C = |g| ∈ [0, 5].  Structural dealer influence independent of sign."""
        return abs(self.g)

    @property
    def direction_dealer_alignment(self) -> float:
        """
        A_DG = d · g ∈ [−25, 25].

        Positive  → direction and dealer flow reinforce each other
                    (smooth trend continuation or stable range).
        Negative  → direction and dealer flow oppose each other
                    (trend friction, unstable transition, or pinned trend).
        Zero      → one or both axes are neutral.
        """
        return self.d * self.g

    # ------------------------------------------------------------------
    # Distance from neutrality
    # ------------------------------------------------------------------

    def weighted_distance_from_neutral(
        self,
        wD: float = 1.0,
        wV: float = 1.0,
        wL: float = 1.0,
        wG: float = 1.0,
    ) -> float:
        """
        δ = sqrt( wD·(D−5)² + wV·(V−5)² + wL·(L−5)² + wG·(G−5)² )

        With unit weights the maximum is 10 (at a corner of the cube).
        Small δ  → close to ordinary market conditions.
        Large δ  → extreme regime.
        """
        return math.sqrt(
            wD * self.d**2
            + wV * (self.V - 5.0) ** 2
            + wL * (self.L - 5.0) ** 2
            + wG * self.g**2
        )

    # ------------------------------------------------------------------
    # Transition / velocity
    # ------------------------------------------------------------------

    def transition_magnitude(
        self,
        prev: MarketCoordinate,
        wD: float = 1.0,
        wV: float = 1.0,
        wL: float = 1.0,
        wG: float = 1.0,
    ) -> float:
        """
        R_trans = ||M_t − M_{t-1}||_W

        Measures how far the market moved in state space between two bars.
        Small R_trans → stable regime; large R_trans → transition / instability.
        Call as:  current.transition_magnitude(previous)
        """
        return math.sqrt(
            wD * (self.D - prev.D) ** 2
            + wV * (self.V - prev.V) ** 2
            + wL * (self.L - prev.L) ** 2
            + wG * (self.G - prev.G) ** 2
        )

    # ------------------------------------------------------------------
    # Regime lattice helpers
    # ------------------------------------------------------------------

    def lattice_index(
        self,
        low_bound: float = 3.0,
        high_bound: float = 7.0,
    ) -> tuple[int, int, int, int]:
        """
        Map each coordinate to a bin index in {1, 2, 3}:

            bin 1 (low)  : x ≤ low_bound   (default ≤ 3)
            bin 2 (mid)  : low_bound < x < high_bound
            bin 3 (high) : x ≥ high_bound  (default ≥ 7)

        Returns (i, j, k, l) ∈ {1,2,3}^4 identifying the cell C_ijkl
        in the 81-cell regime lattice Λ = {1,2,3}^4.
        """

        def _bin(x: float) -> int:
            if x <= low_bound:
                return 1
            if x >= high_bound:
                return 3
            return 2

        return (_bin(self.D), _bin(self.V), _bin(self.L), _bin(self.G))

    def lattice_distance(
        self,
        other: MarketCoordinate,
        low_bound: float = 3.0,
        high_bound: float = 7.0,
    ) -> int:
        """
        Manhattan distance between two cells on the 81-cell lattice.

        d_Λ((i,j,k,l), (i',j',k',l')) = |i−i'| + |j−j'| + |k−k'| + |l−l'|

        Distance 0 → same cell.
        Distance 1 → adjacent cells (smooth regime change).
        Distance > 1 → abrupt regime jump.
        Maximum distance = 8 (opposite corners of the lattice).
        """
        ai, aj, ak, al = self.lattice_index(low_bound, high_bound)
        bi, bj, bk, bl = other.lattice_index(low_bound, high_bound)
        return abs(ai - bi) + abs(aj - bj) + abs(ak - bk) + abs(al - bl)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def neutral() -> MarketCoordinate:
        """Return M_0 = (5, 5, 5, 5), the neutral point of the state space."""
        return MarketCoordinate(D=5.0, V=5.0, L=5.0, G=5.0)

    def __repr__(self) -> str:
        return (
            f"MarketCoordinate(D={self.D:.3f}, V={self.V:.3f}, "
            f"L={self.L:.3f}, G={self.G:.3f})"
        )
