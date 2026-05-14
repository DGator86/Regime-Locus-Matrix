# Regime prediction horizon and options-strategy alignment

## What the regime engine is explicitly predicting

The regime engine computes a **one-step-ahead** regime transition probability vector, not a multi-step direct forecast. In both HMM and Markov paths, the core outputs are `*_next_probs`, `*_most_likely_next_state`, and `*_most_likely_next_prob`, all derived from one-step transition math.

- HMM: `next_p = probs @ transition_matrix_t` and writes `hmm_next_probs` / `hmm_most_likely_next_prob`.
- Markov: `next_p = one_step_predictive_probs(probs, calibrated_transition_matrix)` and writes `markov_next_probs` / `markov_most_likely_next_prob`.

This means the strongest explicit predictive signal is at **t+1 bar**.

## Practical horizon translation

Because the model is one-step in bar space, calendar horizon depends on feed timeframe:

- 1-minute bars: strongest regime signal ≈ next 1 minute.
- 5-minute bars: strongest regime signal ≈ next 5 minutes.
- 1-hour bars: strongest regime signal ≈ next hour.
- Daily bars: strongest regime signal ≈ next trading day.

For farther horizons, confidence generally decays unless persistence is high. The code exposes `*_expected_persistence` and transition entropy as reliability clues, which should be treated as a decay monitor rather than a separate long-horizon forecast.

## Options strategy implication

Given the above, strategy tenor should match one-step regime strength:

1. **High one-step confidence + low transition entropy**
   - Favor **short-DTE directional debit spreads** in bull/bear regimes.
   - In current map, this is the `short_dte=True` branch (`0dte_bull_call_spread`, `0dte_bear_put_spread`, `target_dte_max=2`).

2. **High one-step confidence but high-vol/ambiguous direction**
   - Favor **convexity ownership** (short-dated long straddles).
   - In map: `scalp_long_straddle` (`target_dte_max=2`).

3. **Lower confidence / transition-like regimes**
   - Either no-trade / micro-position, or move to slightly longer DTE defined-risk structures.
   - In map: short-DTE transition low-vol returns `no_trade_or_micro_position`; non-short-DTE defaults to 14–45 DTE defined-risk positions.

## Bottom line

- The engine's explicit strong predictive edge is **next bar (t+1)**.
- Therefore, the best direct exploitation is **0DTE–2DTE defined-risk options** when one-step confidence is strong and entropy is low.
- Use longer DTE (14–45) only when you intentionally trade slower structural views rather than the immediate regime-transition edge.
