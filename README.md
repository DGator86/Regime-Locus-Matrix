# Regime Locus Matrix

## Forecasting

### HMM Hybrid Layer

RLM remains a deterministic and explicit options-native engine. The optional HMM layer adds soft probabilities over the standardized factor space (`S_D`, `S_V`, `S_L`, `S_G`) and exposes persistence/transition-aware confidence for downstream sizing and gating.

- `ForecastPipeline` is unchanged for deterministic operation.
- `HybridForecastPipeline` augments output with:
  - `hmm_probs` (smoothed posterior probabilities per state)
  - `hmm_state` (most likely hidden state)
  - `hmm_state_label` (mode `regime_key` label per state)
- ROEE uses HMM confidence to modulate position sizing and optionally gate low-confidence trades.

Example usage:

```bash
python scripts/run_forecast_pipeline.py --use-hmm --hmm-states 6
python scripts/run_roee_pipeline.py --use-hmm --hmm-states 6
python scripts/run_walkforward.py --use-hmm --hmm-states 6
```

This preserves the explicit policy table while adding probabilistic uncertainty handling on top.
