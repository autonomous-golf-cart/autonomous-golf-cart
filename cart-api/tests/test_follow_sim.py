#!/usr/bin/env python3
"""
test_follow_sim.py — exercise PathFollower's tracking-phase steering law with
injected (simulated) GPS, since a stationary cart can never leave the cold-start
"heading" phase live.

No hardware, no actuation. We build a stub Cart whose .gps.latest returns
positions we control, march it along synthetic trajectories, and assert the
proportional steering law produces the right SIGN and sensible MAGNITUDE:

  1. Straight path, cart already moving straight along it -> steer ~ 0.
  2. Path turning LEFT ahead of a straight-heading cart -> steer NEGATIVE
     (left = CCW = negative column command, matching bearing/angle_diff sign).
  3. Path turning RIGHT -> steer POSITIVE.
  4. With turn_slowdown > 0, gas backs off in a sharp turn vs. straight
     cruise; with the default (0) it holds speed.
  5. Cross-track abort fires when the cart is parked far off the path.
  6. Goal arrival fires (done) at the last waypoint.

Run: python3 -m pytest tests/test_follow_sim.py
"""

import math
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartlib import geo
from cartlib.follow import PathFollower, FollowConfig


# --- stub hardware ---------------------------------------------------------
class StubGps:
    def __init__(self):
        self.latest = None

    def place(self, lat, lon, fix_code=4, fix_type="RTK Fixed"):
        self.latest = {"lat": lat, "lon": lon, "fix_code": fix_code,
                       "fix_type": fix_type, "ts": time.time()}


class StubPedals:
    def __init__(self):
        self.telemetry = {}
        self.gas = self.brake = 0.0

    def set_gas(self, v):
        self.gas = v

    def set_brake(self, v):
        self.brake = v

    def stop(self):
        self.gas = 0.0


class StubCart:
    def __init__(self):
        self.gps = StubGps()
        self.pedals = StubPedals()
        self.steering = None  # dry-run; no steering hardware


# --- geometry helpers for building paths -----------------------------------
def offset(origin, east_m, north_m):
    """Inverse of geo.local_xy: place a point east/north metres from origin."""
    lat0 = origin[0]
    dlat = north_m / geo.EARTH_R
    dlon = east_m / (geo.EARTH_R * math.cos(math.radians(lat0)))
    return (lat0 + math.degrees(dlat), origin[1] + math.degrees(dlon))


ORIGIN = (37.4275, -122.1697)  # somewhere on campus; absolute value irrelevant


def straight_path_north(n=12, step=1.0):
    return [offset(ORIGIN, 0.0, i * step) for i in range(n)]


def turning_path(sign, n=20, step=1.0, curve=4.0):
    """Path heading north then curving. sign=+1 right (east), -1 left (west)."""
    pts = []
    east = 0.0
    for i in range(n):
        north = i * step
        # quadratic lateral growth once past the first few metres
        if i > 3:
            east = sign * curve * ((i - 3) / (n - 3)) ** 2
        pts.append(offset(ORIGIN, east, north))
    return pts


def prime_heading(follower, gps, step_m=0.7):
    """Walk the cart up the first ~few metres of the path so the follower
    establishes a northward heading and enters the tracking phase."""
    last = None
    for i in range(8):
        gps.place(*offset(ORIGIN, 0.0, i * step_m))
        last = follower.step()
    return last


def tracking_step(path, **cfg_overrides):
    """Prime a follower on ``path``, park the cart 5 m up, take one step."""
    cart = StubCart()
    cfg = FollowConfig(require_rtk=False, rate_hz=1000, **cfg_overrides)
    f = PathFollower(cart, path, cfg, armed=False)
    prime_heading(f, cart.gps)
    cart.gps.place(*offset(ORIGIN, 0.0, 5.0))
    return f.step()


# --- tests -----------------------------------------------------------------
def test_straight_path_tracks_north_with_no_steer():
    t = tracking_step(straight_path_north())
    assert t["phase"] == "tracking"
    assert t["heading_deg"] is not None
    assert abs(t["heading_deg"]) < 8 or abs(t["heading_deg"] - 360) < 8
    assert abs(t["steer_cmd"]) < 3.0


def test_left_turn_steers_negative():
    t = tracking_step(turning_path(-1))
    assert t["phase"] == "tracking"
    assert t["alpha"] < -1, "target should be left of heading"
    assert t["steer_cmd"] < -1


def test_right_turn_steers_positive():
    t = tracking_step(turning_path(+1))
    assert t["phase"] == "tracking"
    assert t["alpha"] > 1, "target should be right of heading"
    assert t["steer_cmd"] > 1


def test_default_turn_slowdown_holds_speed():
    straight = tracking_step(straight_path_north())
    turn = tracking_step(turning_path(+1))
    assert turn["gas"] == pytest.approx(straight["gas"])


def test_turn_slowdown_backs_off_gas_in_turn():
    straight = tracking_step(straight_path_north(), turn_slowdown=2.0)
    turn = tracking_step(turning_path(+1), turn_slowdown=2.0)
    assert turn["gas"] < straight["gas"]


def test_crosstrack_abort_when_far_off_path():
    path = straight_path_north()
    cart = StubCart()
    f = PathFollower(cart, path, FollowConfig(max_crosstrack_m=6.0), armed=False)
    prime_heading(f, cart.gps)
    cart.gps.place(*offset(ORIGIN, 20.0, 5.0))  # 20 m east of a due-north path
    t = f.step()
    assert t["phase"] == "abort", f.state.reason


def test_done_at_goal():
    path = straight_path_north()
    cart = StubCart()
    f = PathFollower(cart, path, FollowConfig(goal_radius_m=1.5), armed=False)
    prime_heading(f, cart.gps)
    cart.gps.place(*path[-1])  # sit on the last waypoint
    t = f.step()
    assert t["phase"] == "done", f.state.reason


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
