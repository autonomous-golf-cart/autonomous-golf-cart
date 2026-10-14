#!/usr/bin/env python3
"""
drive.py — drive the cart from the segmentation planner's state file.

!!! WITH --i-understand-this-drives THIS DRIVES THE CART (steering + throttle). !!!

segmentation_infer.py writes steer_deg / target_gas / target_brake to
/tmp/autoware_state.json. This loop reads it at 50 Hz and sends the commands
to the cart through cartlib. It replaces ``ps5_drive.py --autosteer`` from the
old PRODUCTION repo and keeps its arbitration:

  * A state is fresh when its ``ts`` is < 0.30 s old and ``inference`` is true.
  * Steering column = steer_deg * STEER_SIGN (-1, same as the old
    AUTOSTEER_SIGN), clamped to ±270°.
  * Gas and brake are the planner's pot targets, clamped to --max-gas and
    BRAKE_POT_MAX.

Unlike the old script there is no PS5 operator to take over, so a stale or
missing state releases the gas at once and applies --stale-brake once it has
been stale for STALE_BRAKE_AFTER_S. Driving resumes when fresh states return.
Keep a hand on the hardware e-stop.

The steering angle is relative to where the wheel is at startup, so start
with the wheels straight.

Usage (from the repo root, with segmentation_infer.py already running):
    python3 policies/segmentation/onboard/drive.py                              # dry run: print commands only
    python3 policies/segmentation/onboard/drive.py --i-understand-this-drives   # DRIVE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "cart-api"))

from cartlib import Cart, config  # noqa: E402

STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")
CONTROL_HZ = 50.0
STATE_FRESH_S = 0.30        # same as ps5_drive AUTOSTEER_FRESH_S
STALE_BRAKE_AFTER_S = 1.0   # coast this long on a stale state, then brake
STEER_SIGN = -1.0           # planner steer_deg → cart column angle (ps5_drive AUTOSTEER_SIGN)
STEER_MAX_DEG = 270.0       # ps5_drive PS5_STEERING_MAX_DEG
PRINT_HZ = 5.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_state(path: Path) -> dict | None:
    """Latest planner state, or None if missing, stale, or not inferring."""
    try:
        with path.open() as f:
            data = json.load(f)
        age = time.time() - float(data.get("ts", 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if age >= STATE_FRESH_S or not data.get("inference", False):
        return None
    return data


def commands_from_state(state: dict, max_gas: float) -> tuple[float, float, float]:
    """(steer_deg, gas, brake) for the cart from one planner state."""
    try:
        steer = clamp(float(state.get("steer_deg", 0.0)) * STEER_SIGN, -STEER_MAX_DEG, STEER_MAX_DEG)
        gas = clamp(float(state.get("target_gas", 0.0)), 0.0, max_gas)
        brake = clamp(float(state.get("target_brake", 0.0)), 0.0, config.BRAKE_POT_MAX)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0
    return steer, gas, brake


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-this-drives", action="store_true",
                    help="actually drive the cart (default: dry run, no hardware)")
    ap.add_argument("--state-file", type=Path, default=STATE_FILE_DEFAULT)
    ap.add_argument("--max-gas", type=float, default=config.FSD_GAS_LIMIT,
                    help=f"gas ceiling in pot units (default FSD_GAS_LIMIT={config.FSD_GAS_LIMIT})")
    ap.add_argument("--stale-brake", type=float, default=0.15,
                    help="brake pot applied while the planner state is stale")
    args = ap.parse_args()

    live = args.i_understand_this_drives
    max_gas = config.effective_gas_cap(args.max_gas)
    print(f"Mode: {'### LIVE DRIVE ###' if live else 'dry run (no hardware)'}  "
          f"state={args.state_file}  max_gas={max_gas:.3f}  stale_brake={args.stale_brake:.2f}")

    print("Waiting for a fresh planner state (is segmentation_infer.py running?)...")
    while read_state(args.state_file) is None:
        time.sleep(0.1)

    cart = Cart(use_gps=False, gas_cap=max_gas) if live else None
    if cart:
        cart.open()
    try:
        if cart:
            print("Arming pedals + enabling steering...")
            cart.arm()
            if not cart.steering.enable():
                print("Steering failed to enter closed-loop; aborting.")
                return 1

        period = 1.0 / CONTROL_HZ
        steer = 0.0
        last_fresh = time.monotonic()
        next_print = 0.0
        while True:
            now = time.monotonic()
            state = read_state(args.state_file)
            if state is not None:
                last_fresh = now
                steer, gas, brake = commands_from_state(state, max_gas)
                status = "driving"
            else:
                # Hold the last steering angle; coast, then brake.
                gas = 0.0
                stale_s = now - last_fresh
                brake = args.stale_brake if stale_s >= STALE_BRAKE_AFTER_S else 0.0
                status = f"STALE {stale_s:4.1f}s"

            if cart:
                cart.pedals.set_gas(gas)
                cart.pedals.set_brake(brake)
                cart.steering.set_angle(steer)

            if now >= next_print:
                next_print = now + 1.0 / PRINT_HZ
                sys.stdout.write(f"\r[{status:12}] steer={steer:+7.1f}°  gas={gas:.3f}  brake={brake:.3f}   ")
                sys.stdout.flush()

            time.sleep(max(0.0, period - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if cart:
            cart.pedals.set_gas(0.0)
            cart.pedals.set_brake(args.stale_brake)
            time.sleep(0.5)
            cart.close()   # releases pedals and idles the steering motor
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
