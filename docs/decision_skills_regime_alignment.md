# Decision-Making Skills Audit: Equities + Options vs Anticipated Regimes

## Scope

This note traces where the system explicitly converts anticipated regime state into strategy decisions, and how those decisions are validated or overridden before execution.

## 1) Regime-conditioned options strategy selection (core deterministic policy)

Primary router: `get_strategy_for_regime(...)` in `src/rlm/roee/strategy_map.py`.

- Input regime axes: `direction`, `volatility`, `liquidity`, `dealer_flow`.
- Output: a `TradeCandidate` with strategy name, DTE window, profit target, risk cap, and strike geometry (`long_sigma`, `short_sigma`, etc.).
- This is the direct mapping layer from anticipated regime → options structure.

Examples:

- **Bull + low vol + high liquidity + supportive flow** → `long_call_spread` (defined-risk debit spread).
- **Bear + low vol + high liquidity + supportive flow** → `long_put_spread`.
- **Range + high vol** → `long_straddle` (or short-DTE `scalp_long_straddle` when enabled).
- **Transition + low vol in short-DTE mode** → explicit `no_trade_or_micro_position` fallback.

Short-DTE branch (`short_dte=True`) is explicitly separated and uses 0DTE/1DTE logic, favoring tight risk and fast delta/gamma expressions.

## 2) Decision object model ensures strategy intent is machine-readable

`src/rlm/types/options.py` defines:

- `TradeCandidate`: strategy intent + risk/target envelopes + strike placement hints.
- `TradeDecision`: final action (`enter/hold/skip/exit`) plus selected strategy and metadata.

This allows regime-derived decisions to stay structured all the way into downstream plan creation and execution policies.

## 3) Policy engine: regime → strategy name → candidate → sizing/risk gates

In `src/rlm/roee/policy.py`:

- `resolve_strategy_name(...)` calls `get_strategy_for_regime(...)`.
- `_core_trade_decision(...)` converts regime scores and forecasts into a `TradeDecision`.
- `build_candidate_from_strategy_name(...)` supports both coordinate-router strategy names and legacy regime map names, while preserving risk templates.

Net effect: anticipated regime does not just choose a direction; it selects a full options expression and then passes through confidence/sizing/risk logic.

## 4) Universe pipeline links forecasted regime to active plans

`scripts/run_universe_options_pipeline.py` integrates end-to-end flow:

1. Bars/features
2. Forecast/regime stack
3. ROEE decision (`select_trade_for_row`)
4. Strategy/legs packaging into `universe_trade_plans.json`

So the strategy emitted in active plans is downstream of the anticipated regime + confirmation + safety checks, not ad-hoc discretionary selection.

## 5) Equities vs options execution posture

Options execution policy intentionally avoids live IBKR option order placement in this stack (`src/rlm/execution/options_ibkr_policy.py` and `src/rlm/execution/ibkr_combo_orders.py`), while equities can be traded via dedicated IBKR equity flows.

Interpretation:

- **Decision intelligence is options-first and regime-driven**.
- **Live broker execution path is equity-first in current stack defaults**.

This means regime-skill quality should be evaluated separately from broker plumbing constraints.

## 6) Hermes “decision-making skills” layer (meta-decision / operator intelligence)

Hermes skills add a second decision layer:

- `hermes_skills/research_analyst/SKILL.md`: requires regime context tool + optional TradingAgents conviction checks per symbol.
- `hermes_skills/commander/SKILL.md`: synthesizes pipeline health + regime research + TradingAgents alignment/conflict into system posture (`AGGRESSIVE/NORMAL/DEFENSIVE/STAND-DOWN`) and command decision (`GO/HOLD/STAND-DOWN/ALERT`).
- `hermes_skills/data_monitor/SKILL.md`: health-first constraints to prevent false urgency during market-closed windows.

The registry in `src/rlm_hermes_tools/register_rlm_tools.py` wires these skills to factual tools:

- `rlm_get_trade_and_regime_context`
- `rlm_get_system_gate_state`
- `rlm_check_portfolio_limits`
- `rlm_get_trading_agents_analysis`

## 7) Strengths and direct correlation quality

Strong explicit correlation exists because:

- Strategy mapping is deterministic and regime-keyed.
- Each mapped strategy embeds risk and expiry assumptions appropriate to regime semantics.
- No-trade branches are explicitly encoded for low-edge ambiguous states.
- Commander/research skills can veto or downgrade action when cross-signal conflict appears.

## 8) Practical caveat for evaluation

When evaluating “equities and options strategy choice quality,” separate:

1. **Regime-to-strategy decision quality** (implemented and explicit), from
2. **Live execution channel limitations** (options mostly tracked/simulated; equities executable through IBKR paths).

