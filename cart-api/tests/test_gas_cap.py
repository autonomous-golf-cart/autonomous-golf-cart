"""
test_gas_cap.py — the layered gas-authority hierarchy in cartlib.config.

effective_gas_cap() must never return more than the hardware ceiling or the
global governor, whatever a mode asks for.

Run: python3 -m pytest tests/
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartlib import config


def test_layers_are_ordered():
    assert config.FSD_GAS_LIMIT <= config.GLOBAL_SPEED_LIMIT <= config.GAS_POT_MAX


def test_mode_limit_below_every_layer_passes_through():
    assert config.effective_gas_cap(0.1) == pytest.approx(0.1)


def test_mode_limit_clamped_by_global_governor():
    assert config.effective_gas_cap(config.GAS_POT_MAX) == pytest.approx(config.GLOBAL_SPEED_LIMIT)


def test_huge_mode_limit_never_exceeds_hardware_or_governor():
    cap = config.effective_gas_cap(10.0)
    assert cap <= config.GAS_POT_MAX
    assert cap <= config.GLOBAL_SPEED_LIMIT


def test_fsd_cap_is_fsd_limit():
    assert config.effective_gas_cap(config.FSD_GAS_LIMIT) == pytest.approx(config.FSD_GAS_LIMIT)


def test_zero_mode_limit_means_no_gas():
    assert config.effective_gas_cap(0.0) == 0.0


def test_gas_for_mph_zero_or_negative_is_zero():
    assert config.gas_for_mph(0.0) == 0.0
    assert config.gas_for_mph(-3.0) == 0.0


def test_gas_for_mph_low_speed_hits_rolling_floor():
    assert config.gas_for_mph(1.0) == pytest.approx(config.ROLLING_GAS_FLOOR)


def test_gas_for_mph_calibration_point():
    assert config.gas_for_mph(config.GAS_CALIBRATION_MPH) == pytest.approx(config.GAS_CALIBRATION_POT)


def test_gas_for_mph_is_capped():
    assert config.gas_for_mph(100.0) == pytest.approx(config.effective_gas_cap(10.0))


def test_config_reexports_shared_limits():
    assert config.effective_gas_cap is config._limits.effective_gas_cap
    assert config.FSD_GAS_LIMIT == config._limits.FSD_GAS_LIMIT
