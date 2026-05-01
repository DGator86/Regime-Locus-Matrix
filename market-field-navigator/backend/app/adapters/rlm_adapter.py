from __future__ import annotations


class RLMAdapter:
    def __init__(self, repo_root: str | None = None):
        self.repo_root = repo_root

    def get_snapshot_inputs(self, symbol: str) -> dict:
        return {
            "symbol": symbol.upper(),
            "current_price": 5071.25,
            "anchor_price": 5046.50,
            "regime_label": "Bull Drive",
            "regime_probabilities": {"bull": 0.72, "bear": 0.11, "chop": 0.17},
            "hmm_state": "trend_continuation",
            "transition_probabilities": {
                "trend_continuation": 0.64,
                "mean_reversion": 0.21,
                "chop": 0.15,
            },
            "top_drivers": [
                {
                    "name": "Gamma Support",
                    "direction": "positive",
                    "weight": 0.76,
                    "explanation": "Dealer flow is supportive above current price.",
                },
                {
                    "name": "Resistance Nearby",
                    "direction": "negative",
                    "weight": 0.58,
                    "explanation": "Price is approaching a major resistance wall.",
                },
            ],
            "confidence": 0.68,
            "recommended_action": "WAIT_FOR_CONFIRMATION",
            "risk_state": "constructive_but_near_resistance",
            "levels": {"support": [4950, 4820], "resistance": [5210, 5320]},
        }
