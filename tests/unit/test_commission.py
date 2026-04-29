"""Tests for the commission model."""

from __future__ import annotations

import pytest

from rlm.backtest.commission import (
    CommissionConfig,
    CommissionModel,
    calculate_commission,
)


class TestCalculateCommission:
    def test_per_contract_model(self) -> None:
        config = CommissionConfig(model=CommissionModel.PER_CONTRACT, per_contract_rate=0.65)
        # 2 legs, 1 contract each
        assert calculate_commission(config=config, leg_count=2, quantity=1) == pytest.approx(1.30)

    def test_per_contract_multiple_quantity(self) -> None:
        config = CommissionConfig(model=CommissionModel.PER_CONTRACT, per_contract_rate=0.65)
        # 4 legs, 3 contracts each = 12 contracts total
        assert calculate_commission(config=config, leg_count=4, quantity=3) == pytest.approx(7.80)

    def test_per_trade_model(self) -> None:
        config = CommissionConfig(model=CommissionModel.PER_TRADE, per_trade_rate=5.0)
        # Flat fee regardless of leg/quantity count
        assert calculate_commission(config=config, leg_count=4, quantity=10) == pytest.approx(5.0)

    def test_per_trade_model_single_leg(self) -> None:
        config = CommissionConfig(model=CommissionModel.PER_TRADE, per_trade_rate=2.50)
        assert calculate_commission(config=config, leg_count=1, quantity=1) == pytest.approx(2.50)

    def test_hybrid_model(self) -> None:
        config = CommissionConfig(
            model=CommissionModel.HYBRID,
            per_trade_rate=1.0,
            per_contract_rate=0.50,
        )
        # Base 1.0 + (2 legs * 1 contract * 0.50) = 2.0
        assert calculate_commission(config=config, leg_count=2, quantity=1) == pytest.approx(2.0)

    def test_hybrid_model_multiple_quantity(self) -> None:
        config = CommissionConfig(
            model=CommissionModel.HYBRID,
            per_trade_rate=1.0,
            per_contract_rate=0.50,
        )
        # Base 1.0 + (4 legs * 2 quantity * 0.50) = 1.0 + 4.0 = 5.0
        assert calculate_commission(config=config, leg_count=4, quantity=2) == pytest.approx(5.0)

    def test_min_commission_floor(self) -> None:
        config = CommissionConfig(
            model=CommissionModel.PER_CONTRACT,
            per_contract_rate=0.10,
            min_commission=1.0,
        )
        # 0.10 * 1 contract = 0.10, but floor is 1.0
        assert calculate_commission(config=config, leg_count=1, quantity=1) == pytest.approx(1.0)

    def test_min_commission_not_applied_above_floor(self) -> None:
        config = CommissionConfig(
            model=CommissionModel.PER_CONTRACT,
            per_contract_rate=0.65,
            min_commission=0.50,
        )
        # 0.65 * 2 = 1.30, above the 0.50 floor
        assert calculate_commission(config=config, leg_count=2, quantity=1) == pytest.approx(1.30)

    def test_from_legacy_rate(self) -> None:
        config = CommissionConfig.from_legacy_rate(0.65)
        assert config.model == CommissionModel.PER_CONTRACT
        assert config.per_contract_rate == pytest.approx(0.65)
        assert calculate_commission(config=config, leg_count=1, quantity=1) == pytest.approx(0.65)

    def test_zero_commission(self) -> None:
        config = CommissionConfig(model=CommissionModel.PER_CONTRACT, per_contract_rate=0.0)
        assert calculate_commission(config=config, leg_count=4, quantity=5) == 0.0
