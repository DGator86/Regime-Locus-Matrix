# Error Audit (2026-05-14)

## Scope
- Ran full unit/integration test suite (`pytest tests/`).
- Ran lint checks (`ruff check src/ tests/`).
- Ran formatting check (`black --check --target-version py312 src/ tests/`).
- Ran static type checks (`mypy src/`).

## Failures found

### Test failures (2)
1. `tests/unit/test_optimization_imports.py::test_microstructure_collector_compatibility_cli_modules_show_help`
   - Subprocess `python -m rlm.microstructure.collectors.options --help` fails with `ModuleNotFoundError: No module named 'rlm'`.
2. `tests/unit/test_startup_decision_tree_health.py::test_startup_health_script_exits_zero`
   - `scripts/run_startup_decision_tree_health.py` raises `json.decoder.JSONDecodeError` while reading `spy_large_plan.json`.

### Lint failures (38)
- Major blocking set is in `src/rlm/forecasting/hmm.py`: method redefinitions, undefined names (`gamma`, `current`, `calibrated`, `eta`).
- Additional lint failures include import ordering, unused imports/variables, one trailing whitespace, and one extraneous f-string.

### Type-check failures (12)
- `src/rlm/forecasting/models/kronos/model/kronos.py`: symbol redefinitions (`KronosTokenizer`, `Kronos`, `KronosPredictor`).
- `src/rlm/forecasting/hmm.py`: duplicate method definitions and undefined names, matching ruff findings.

### Formatting check failures
- `black --check` reports 31 files needing reformat.

## Notes
- This audit records currently reproducible failures and does not attempt fixes.
