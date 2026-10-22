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

Run: python3 tests/test_follow_sim.py
"""

import math
import os
import sys
import time

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


# --- test driver -----------------------------------------------------------
PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def prime_heading(follower, gps, path, step_m=0.7):
    """Walk the cart up the first ~few metres of the path so the follower
    establishes a northward heading and enters the tracking phase."""
    last = None
    for i in range(8):
        p = offset(ORIGIN, 0.0, i * step_m)
        gps.place(*p)
        tele = follower.step()
        last = tele
    return last


def run_case(name, path, expect, **cfg_overrides):
    cart = StubCart()
    cfg = FollowConfig(require_rtk=False, rate_hz=1000, **cfg_overrides)
    f = PathFollower(cart, path, cfg, armed=False)
    tele = prime_heading(f, cart.gps, path)

    # Now park the cart at the start region and take one tracking step.
    cart.gps.place(*offset(ORIGIN, 0.0, 5.0))
    tele = f.step()
    print(f"\n{name}: phase={tele['phase']} heading={tele['heading_deg']} "
          f"alpha={tele['alpha']} steer={tele['steer_cmd']} gas={tele['gas']}")
    expect(tele)
    return tele


def main():
    print("PathFollower tracking-phase simulation\n" + "=" * 42)

    # 1. heading establishes + straight tracking ~0 steer
    def straight_expect(t):
        check("enters tracking phase", t["phase"] == "tracking", t["phase"])
        check("heading ~ north (0deg)",
              t["heading_deg"] is not None and (abs(t["heading_deg"]) < 8 or abs(t["heading_deg"] - 360) < 8),
              str(t["heading_deg"]))
        check("steer near zero on straight", abs(t["steer_cmd"]) < 3.0, str(t["steer_cmd"]))
    straight = run_case("STRAIGHT", straight_path_north(), straight_expect)

    # 2. left turn -> negative steer
    def left_expect(t):
        check("tracking", t["phase"] == "tracking", t["phase"])
        check("alpha negative (target left of heading)", t["alpha"] < -1, str(t["alpha"]))
        check("steer NEGATIVE for left turn", t["steer_cmd"] < -1, str(t["steer_cmd"]))
    left = run_case("LEFT TURN", turning_path(-1), left_expect)

    # 3. right turn -> positive steer
    def right_expect(t):
        check("tracking", t["phase"] == "tracking", t["phase"])
        check("alpha positive (target right of heading)", t["alpha"] > 1, str(t["alpha"]))
        check("steer POSITIVE for right turn", t["steer_cmd"] > 1, str(t["steer_cmd"]))
    right = run_case("RIGHT TURN", turning_path(+1), right_expect)

    # 4. turn_slowdown defaults to 0 (hold speed); when set, gas backs off
    print()
    check("default turn_slowdown holds speed in turn",
          right["gas"] == straight["gas"],
          f"turn={right['gas']} straight={straight['gas']}")
    slowed = run_case("RIGHT TURN (turn_slowdown=2)", turning_path(+1), right_expect,
                      turn_slowdown=2.0)
    check("gas backed off in turn with turn_slowdown",
          slowed["gas"] < straight["gas"],
          f"turn={slowed['gas']} straight={straight['gas']}")

    # 5. cross-track abort: park far off the path
    print("\nCROSS-TRACK ABORT:")
    cart = StubCart()
    f = PathFollower(cart, straight_path_north(), FollowConfig(max_crosstrack_m=6.0), armed=False)
    prime_heading(f, cart.gps, straight_path_north())
    cart.gps.place(*offset(ORIGIN, 20.0, 5.0))  # 20 m east of a due-north path
    t = f.step()
    check("aborts when off path", t["phase"] == "abort", f"{t['phase']} / {f.state.reason}")

    # 6. goal arrival
    print("\nGOAL ARRIVAL:")
    path = straight_path_north()
    cart = StubCart()
    f = PathFollower(cart, path, FollowConfig(goal_radius_m=1.5), armed=False)
    prime_heading(f, cart.gps, path)
    cart.gps.place(*path[-1])  # sit on the last waypoint
    t = f.step()
    check("reports done at goal", t["phase"] == "done", f"{t['phase']} / {f.state.reason}")

    print("\n" + "=" * 42)
    ok = all(results)
    print(f"{sum(results)}/{len(results)} checks passed -> "
          f"{'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
