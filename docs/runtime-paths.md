# Runtime paths

RLM resolves data roots in this order:
1. `--data-root`
2. `RLM_DATA_ROOT`
3. `./data`

Run manifests are written to `artifacts/runs/` under the selected data root.
