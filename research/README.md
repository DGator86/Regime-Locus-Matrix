# research/

Experimental notebooks, fine-tuning scripts, and one-off analyses.

These are **not** part of the stable public surface.  APIs and outputs may
change without notice.

## Contents

| Path | Description |
|------|-------------|
| `notebooks/` | Jupyter notebooks (regime stratification, Kronos fine-tune, etc.) |
| `finetune_kronos.py` | Fine-tune the Kronos foundation model on local bar data |
| `upload_kronos_checkpoints_hf.py` | Push fine-tuned checkpoints to HuggingFace Hub |
| `train_probabilistic_model.py` | Train quantile-regression probabilistic model |
| `train_coordinate_models.py` | Train coordinate regime models |
| `optimize_forecast_params.py` | Optuna-based forecast hyperparameter search |
| `weekly_regime_model_tournament.sh` | Cron-friendly weekly model tournament |

## Research edge ideas

1. **Regime-conditional factor weighting**
   - Replace global/static coefficients with a state-aware linear layer so each HMM state has its own factor weights.
   - Example hypothesis: in high-volatility states, GEX and orderflow features carry more signal; in low-volatility states, momentum/term-structure factors may dominate.
   - Practical implementation path: fit per-state ridge/elastic-net models (or one model with regime-interaction terms), then evaluate lift vs. the current shared-weight baseline in walk-forward tests.

2. **Reinforcement learning for ROEE policy**
   - Treat ROEE as an MDP where observation = factor vector + regime context, and action = sizing/entry/hold/exit decision.
   - Use historical pipeline outputs as an offline RL dataset (state, action, reward, next-state), with existing PnL/risk metrics as reward components.
   - Keep the current Kelly + gating policy as the benchmark and safety fallback; evaluate RL candidates under strict out-of-sample and turnover/slippage constraints.

