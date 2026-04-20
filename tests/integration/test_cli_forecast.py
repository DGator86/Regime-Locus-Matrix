import argparse

from rlm.cli.common import build_pipeline_config


def test_build_pipeline_config_with_profile_symbol():
    args = argparse.Namespace(
        use_hmm=False,
        use_markov=False,
        hmm_states=None,
        markov_states=None,
        probabilistic=False,
        model_path=None,
        no_kronos=False,
        no_vix=False,
        run_backtest=False,
        initial_capital=None,
        profile="forecast",
        config=None,
    )
    cfg = build_pipeline_config(args, "SPY")
    assert cfg.symbol == "SPY"
