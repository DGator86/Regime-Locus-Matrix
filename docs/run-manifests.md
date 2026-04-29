# Run manifests

Major commands write `artifacts/runs/<run_id>.json` with config, inputs, outputs, and metrics.

Backtest manifests now persist the full `FullRLMConfig` payload under `config_summary.full_rlm_config`, so every run captures the exact configuration used and can be diffed later.
