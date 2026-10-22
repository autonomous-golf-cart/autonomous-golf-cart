"""
test_steering_conversions.py — steering-column degrees <-> ODrive motor turns
through the 3:1 HTD-5M belt, plus the heading-offset scaling.

Run: python3 -m pytest tests/
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartlib import config


def test_belt_ratio():
    assert config.STEERING_BELT_RATIO == 3.0


def test_full_column_turn_is_three_motor_turns():
    assert config.steering_deg_to_motor_turns(360.0) == pytest.approx(3.0)


def test_center_is_zero():
    assert config.steering_deg_to_motor_turns(0.0) == 0.0
    assert config.motor_turns_to_steering_deg(0.0) == 0.0


def test_sign_is_preserved():
    assert config.steering_deg_to_motor_turns(-90.0) == pytest.approx(-0.75)
    assert config.motor_turns_to_steering_deg(-0.75) == pytest.approx(-90.0)


def test_one_motor_turn_is_120_degrees():
    assert config.motor_turns_to_steering_deg(1.0) == pytest.approx(120.0)


@pytest.mark.parametrize("deg", [-320.0, -137.5, -1.0, 0.0, 12.25, 90.0, 320.0])
def test_round_trip(deg):
    turns = config.steering_deg_to_motor_turns(deg)
    assert config.motor_turns_to_steering_deg(turns) == pytest.approx(deg)


def test_soft_limits_symmetric():
    assert config.STEERING_MIN_DEG == -config.STEERING_MAX_DEG


def test_heading_offset_full_lock():
    assert config.steer_cmd_to_heading_offset(config.STEERING_MAX_DEG) == pytest.approx(
        config.HEADING_FULL_LOCK_DEG
    )
    assert config.steer_cmd_to_heading_offset(config.STEERING_MIN_DEG) == pytest.approx(
        -config.HEADING_FULL_LOCK_DEG
    )


def test_heading_offset_is_linear():
    half = config.steer_cmd_to_heading_offset(config.STEERING_MAX_DEG / 2)
    assert half == pytest.approx(config.HEADING_FULL_LOCK_DEG / 2)
