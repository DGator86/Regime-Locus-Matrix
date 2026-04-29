import math

from rlm.roee.policy import select_trade


def _dynamic_sizing_kwargs() -> dict[str, float | bool]:
    return {
        "forecast_return": 1.0,
        "realized_vol": 1.5,
        "use_dynamic_sizing": True,
        "max_kelly_fraction": 0.25,
    }


def test_roee_selects_bull_call_spread_for_clean_bull_state() -> None:
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
    )

    assert decision.action == "enter"
    assert decision.strategy_name == "long_call_spread"
    assert len(decision.legs) == 2
    assert decision.size_fraction is not None
    assert decision.size_fraction > 0.0


def test_roee_skips_major_event() -> None:
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        has_major_event=True,
        strike_increment=5.0,
    )

    assert decision.action == "skip"


def test_roee_skips_transition_probe_default() -> None:
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.35,
        s_v=0.0,
        s_l=0.2,
        s_g=0.1,
        direction_regime="transition",
        volatility_regime="transition",
        liquidity_regime="low_liquidity",
        dealer_flow_regime="destabilizing",
        regime_key="transition|transition|low_liquidity|destabilizing",
        strike_increment=5.0,
    )

    assert decision.action == "skip"
    assert decision.strategy_name == "no_trade_or_micro_position"


def test_roee_builds_iron_condor_for_clean_range_state() -> None:
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.0,
        s_v=-0.5,
        s_l=0.8,
        s_g=0.7,
        direction_regime="range",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="range|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
    )

    assert decision.action == "enter"
    assert decision.strategy_name == "long_iron_condor"
    assert len(decision.legs) == 4


def test_roee_regime_adjusted_kelly_cuts_size_in_high_vol_markov_state() -> None:
    baseline = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        regime_adjusted_kelly=False,
        **_dynamic_sizing_kwargs(),
    )
    stressed = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        regime_state_label="bear|high_vol|high_liquidity|supportive_like",
        regime_state_confidence=0.8,
        **_dynamic_sizing_kwargs(),
    )

    assert baseline.size_fraction == 0.018
    assert stressed.size_fraction == 0.0108
    assert float(stressed.metadata["max_kelly_fraction"]) == 0.15
    assert float(stressed.metadata["kelly_fraction_multiplier"]) == 0.6


def test_roee_regime_adjusted_kelly_boosts_size_in_calm_trending_markov_state() -> None:
    baseline = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        regime_adjusted_kelly=False,
        **_dynamic_sizing_kwargs(),
    )
    accelerated = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        regime_state_label="bull|low_vol|high_liquidity|supportive_like",
        regime_state_confidence=0.8,
        **_dynamic_sizing_kwargs(),
    )

    assert baseline.size_fraction == 0.018
    assert accelerated.size_fraction == 0.0216
    assert float(accelerated.metadata["max_kelly_fraction"]) == 0.3
    assert float(accelerated.metadata["kelly_fraction_multiplier"]) == 1.2


def test_dynamic_kelly_sizing_treats_nan_uncertainty_as_missing() -> None:
    baseline = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=None,
        vault_uncertainty_threshold=None,
        **_dynamic_sizing_kwargs(),
    )
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=float("nan"),
        vault_uncertainty_threshold=None,
        **_dynamic_sizing_kwargs(),
    )

    assert baseline.size_fraction == 0.018
    assert decision.size_fraction == baseline.size_fraction
    assert not math.isnan(float(decision.metadata["kelly_confidence_multiplier"]))


def test_roee_vault_halves_size_when_uncertainty_exceeds_threshold() -> None:
    baseline = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=0.06,
        vault_uncertainty_threshold=None,
    )
    vaulted = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=0.06,
        vault_uncertainty_threshold=0.03,
        vault_size_multiplier=0.5,
    )

    assert baseline.action == "enter"
    assert vaulted.action == "enter"
    assert baseline.size_fraction is not None
    assert vaulted.size_fraction == baseline.size_fraction * 0.5
    assert vaulted.metadata["vault_triggered"] is True
    assert vaulted.metadata["vault_uncertainty_threshold"] == 0.03
    assert vaulted.metadata["forecast_uncertainty"] == 0.06


def test_roee_vault_leaves_size_unchanged_below_threshold() -> None:
    baseline = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=0.02,
        vault_uncertainty_threshold=None,
    )
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=5.0,
        forecast_uncertainty=0.02,
        vault_uncertainty_threshold=0.03,
        vault_size_multiplier=0.5,
    )

    assert baseline.action == "enter"
    assert decision.action == "enter"
    assert decision.size_fraction == baseline.size_fraction
    assert decision.metadata["vault_triggered"] is False
