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
