# Persona Decision Pipeline

A lightweight, deterministic four-stage interpretation layer that sits on top of existing RLM pipeline outputs.

## Overview

The persona pipeline reads the last bar of a `PipelineResult` and produces a structured `PersonaPipelineResult` — a compact trade-interpretation artefact with bias, trap assessment, a directive, and an audit note.

```
FullRLMPipeline.run(bars_df)
        │
        ▼
PersonaDecisionPipeline.run(result)
        │
        ├─ Seven  → signal_alignment, confidence, bias
        ├─ Garak  → trap_risk, dealer_alignment, veto
        ├─ Sisko  → directive (long | short | no_trade) + execution policies
        └─ Data   → historical_edge, regime_match, adaptation_note
```

## Stages

| Stage | Responsibility | Key outputs |
|-------|----------------|-------------|
| **Seven** | Signal normalisation — reads `S_D`, `S_L`, `S_G`, `direction_regime`, `hmm_confidence` | `bias`, `signal_alignment`, `confidence` |
| **Garak** | Trap / deception detection — combines vol stress, liquidity quality, dealer opposition | `trap_risk`, `dealer_alignment`, `veto` |
| **Sisko** | Final directive authority — gates on Garak veto, confidence, and alignment | `directive` (`long`/`short`/`no_trade`), execution policies |
| **Data**  | Empirical audit — uses backtest metrics when available, otherwise factor-score proxy | `regime_match`, `historical_edge`, `adaptation_note`, `review_flag` |

## Output shape

```json
{
  "seven": {
    "bias": "bullish",
    "signal_alignment": 0.81,
    "confidence": 0.74
  },
  "garak": {
    "trap_risk": 0.22,
    "dealer_alignment": "supportive",
    "liquidity_comment": "breakout appears clean with supportive liquidity",
    "veto": false
  },
  "sisko": {
    "directive": "long",
    "entry_policy": "take breakout continuation on confirmed momentum",
    "invalidation_policy": "fail back below trigger zone invalidates setup",
    "target_policy": "scale at first expansion; trail remainder"
  },
  "data": {
    "regime_match": "high",
    "historical_edge": 0.63,
    "adaptation_note": "similar setups perform best in expanding vol; watch for vol expansion trigger",
    "review_flag": false
  }
}
```

## Usage

### From Python

```python
from rlm.core.pipeline import FullRLMPipeline
from rlm.persona import PersonaDecisionPipeline

rlm_result = FullRLMPipeline().run(bars_df)
persona = PersonaDecisionPipeline().run(rlm_result)

print(persona.sisko.directive)      # "long" | "short" | "no_trade"
print(persona.garak.veto)           # True / False
print(persona.seven.confidence)     # float 0–1

import json
print(json.dumps(persona.to_dict(), indent=2))
```

### From CLI

Pass `--persona` to `rlm trade`; the interpretation is printed after the standard decision output:

```bash
rlm trade --symbol SPY --mode plan --persona
```

### With custom thresholds

```python
from rlm.persona import PersonaDecisionPipeline, PersonaConfig

cfg = PersonaConfig(
    trap_risk_veto_threshold=0.55,   # tighter trap filter
    confidence_threshold=0.50,        # stricter confidence gate
)
persona = PersonaDecisionPipeline(cfg).run(rlm_result)
```

## Directive logic

```
garak.veto=True              → no_trade
seven.confidence < 0.40      → no_trade
seven.signal_alignment < 0.55 → no_trade
seven.bias = "bullish"        → long
seven.bias = "bearish"        → short
seven.bias = "neutral"        → no_trade
```

All thresholds live in `PersonaConfig` and can be overridden without touching source code.

## Configuration reference

| Field | Default | Description |
|-------|---------|-------------|
| `signal_alignment_threshold` | 0.55 | Minimum alignment score for a directional directive |
| `confidence_threshold` | 0.45 | Minimum Seven confidence (unused by Sisko — see `min_confidence_for_directional`) |
| `min_confidence_for_directional` | 0.40 | Sisko confidence gate |
| `trap_risk_veto_threshold` | 0.65 | Garak veto trigger |
| `vol_risk_weight` | 0.40 | Weight of vol stress in trap composite |
| `liq_risk_weight` | 0.40 | Weight of liquidity stress in trap composite |
| `dealer_risk_weight` | 0.20 | Weight of opposed dealer flow in trap composite |
| `regime_match_high_threshold` | 0.65 | historical_edge → "high" |
| `regime_match_moderate_threshold` | 0.40 | historical_edge → "moderate" |
| `review_flag_edge_threshold` | 0.42 | Sets `review_flag=True` when edge is below this |

## Design principles

- **No new models** — reads existing `S_D / S_V / S_L / S_G` scores, regime labels, and HMM confidence.
- **No external calls** — fully local, no network, no LLM, no database.
- **Optional** — zero impact on the main pipeline when `use_persona=False` (default).
- **Testable** — all stages are pure functions over `PersonaInputs`; every code path covered by synthetic-fixture unit tests.
- **Configurable** — every threshold in `PersonaConfig`; fits the existing config-override pattern.

## Running the tests

```bash
pytest tests/unit/test_persona_pipeline.py -v
```
