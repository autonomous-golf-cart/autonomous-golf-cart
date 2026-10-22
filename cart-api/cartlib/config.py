"""
cartlib.config — device ports, baud rates, and cart limits.

The mechanical / software limits are not defined here: they are loaded from
the shared ``hardware/arduino-sketches/limits.py`` (mirrored in C by
``sketches/common/cart_limits.h``) and re-exported, so cartlib can't drift
from the numbers the firmware side uses.

Device discovery prefers the stable ``/dev/serial/by-id/`` symlinks so the
library doesn't care whether the GPS, ODrive, and Arduino enumerate as
ttyACM0/1/2 in some other order after a reboot or re-plug.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path


def _load_shared_limits():
    """Import ``hardware/arduino-sketches/limits.py`` from the repo checkout.

    That directory isn't a Python package (its name has a hyphen), so load
    the file directly. cartlib is installed editable (``pip install -e``),
    so the path resolves relative to this source file.
    """
    path = Path(__file__).resolve().parents[2] / "hardware" / "arduino-sketches" / "limits.py"
    spec = importlib.util.spec_from_file_location("_cart_shared_limits", path)
    if spec is None or not path.is_file():
        raise ImportError(f"shared cart limits not found at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_limits = _load_shared_limits()

# --------------------------------------------------------------------------
# Serial port discovery
# --------------------------------------------------------------------------
# Each device is matched by a substring of its /dev/serial/by-id/ name.
# These come straight from `ls /dev/serial/by-id/` on the cart:
#   usb-Linux_..._CDC_Composite_Gadget-if02                  -> GPS bridge
#   usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00  -> GPS
#   usb-ODrive_Robotics_ODrive_S1_0062747288A9-if00           -> steering
#   usb-Arduino__www.arduino.cc__0042_1453130...-if00         -> pedals
_BY_ID_DIR = "/dev/serial/by-id"

GPS_ID_HINTS = ("CDC_Composite_Gadget", "u-blox", "GNSS")
ODRIVE_ID_HINTS = ("ODrive",)
ARDUINO_ID_HINTS = ("Arduino",)

# Fallback raw ports if the by-id symlinks are unavailable for some reason.
# GPS should normally be found by the stable u-blox /dev/serial/by-id link.
# Set FOLLOWRTK_GPS_PORT=/dev/tty... only for intentionally custom wiring.
GPS_FALLBACK = None
GPS_BRIDGE = "/dev/gps-bridge"
ODRIVE_FALLBACK = "/dev/ttyACM1"
ARDUINO_FALLBACK = "/dev/ttyACM2"

# --------------------------------------------------------------------------
# Baud rates
# --------------------------------------------------------------------------
GPS_BAUD = int(os.getenv("FOLLOWRTK_GPS_BAUD", "38400"))  # Pi bridge/u-blox UART baud
ARDUINO_BAUD = 115200   # pedal_control.ino — 115200 8N1, newline terminated
ODRIVE_BAUD = 115200    # ODrive USB-CDC ASCII protocol (baud is nominal over CDC)

# --------------------------------------------------------------------------
# Pedal actuator limits (normalized pot units, 0.0 = released)
#   from hardware/arduino-sketches/limits.py
# --------------------------------------------------------------------------
GAS_POT_MIN = _limits.GAS_POT_MIN
GAS_POT_MAX = _limits.GAS_POT_MAX      # full throttle (actuator mechanical hard stop)
BRAKE_POT_MIN = _limits.BRAKE_POT_MIN
BRAKE_POT_MAX = _limits.BRAKE_POT_MAX  # full brake (actuator mechanical limit)

# Layered gas authority caps (see limits.py for the rationale).
GLOBAL_SPEED_LIMIT = _limits.GLOBAL_SPEED_LIMIT  # top-level governor, applies to everyone
FSD_GAS_LIMIT = _limits.FSD_GAS_LIMIT            # autonomy cap
effective_gas_cap = _limits.effective_gas_cap

# Production open-loop speed calibration. The linear 5-8 mph regime is about
# 0.24 gas pot at 8 mph, but low-speed commands need a floor to overcome
# rolling resistance; otherwise 1-4 mph barely moves the actuator/cart.
GAS_CALIBRATION_MPH = 8.0
GAS_CALIBRATION_POT = 0.24
GAS_PER_MPH = GAS_CALIBRATION_POT / GAS_CALIBRATION_MPH
ROLLING_GAS_FLOOR = 0.18

# The Arduino trips FAILSAFE if it hears nothing for 300 ms. Send a heartbeat
# comfortably faster than that.
PEDAL_HEARTBEAT_HZ = 20
PEDAL_HEARTBEAT_TIMEOUT_MS = 300

# --------------------------------------------------------------------------
# Steering limits (ODrive S1 + M8325s through 3:1 HTD-5M belt)
#   from hardware/arduino-sketches/limits.py
# --------------------------------------------------------------------------
STEERING_BELT_RATIO = _limits.STEERING_BELT_RATIO  # 3 motor turns = 1 steering-column turn
# Steering-column soft limits. The cart rack physically supports ~±320°.
STEERING_MIN_DEG = _limits.STEERING_MIN_DEG
STEERING_MAX_DEG = _limits.STEERING_MAX_DEG
steering_deg_to_motor_turns = _limits.steering_deg_to_motor_turns
motor_turns_to_steering_deg = _limits.motor_turns_to_steering_deg

# Scales the steering "straighten" (heading) term only — NOT the map needle,
# which now leans by the wheel angle 1:1. The controller treats full lock
# (~±320° column) as a ~75° heading deviation so the centering term stays gentle
# instead of saturating. Bump it to make `straighten` bite harder, drop it for
# softer centering.
HEADING_FULL_LOCK_DEG = 75.0


def steer_cmd_to_heading_offset(column_deg: float) -> float:
    """Wheel angle -> scaled heading deviation for the steering centering term."""
    return column_deg * (HEADING_FULL_LOCK_DEG / STEERING_MAX_DEG)

# Conservative trapezoidal-trajectory limits for library-driven steering
# (motor-side units). Production sweeps faster; these are gentle defaults.
STEERING_TRAP_VEL = 4.0     # turns/s
STEERING_TRAP_ACCEL = 8.0   # turns/s^2
STEERING_TRAP_DECEL = 8.0   # turns/s^2



def gas_for_mph(speed_mph: float) -> float:
    """Open-loop pedal pot target for a requested constant speed in mph."""
    if speed_mph <= 0.0:
        return 0.0
    return effective_gas_cap(max(speed_mph * GAS_PER_MPH, ROLLING_GAS_FLOOR))


def mph_from_gas(gas: float) -> float:
    """Best-effort inverse of the linear part of ``gas_for_mph``."""
    return max(0.0, gas / GAS_PER_MPH)


# --------------------------------------------------------------------------
# Port resolution
# --------------------------------------------------------------------------
def _resolve_by_id(hints, fallback, *, allow_any_acm: bool = True):
    """Return the first /dev/serial/by-id link whose name matches a hint.

    Resolves the symlink to its real /dev/ttyACMx target. Falls back to the
    given raw port if nothing matches (e.g. by-id dir missing).
    """
    try:
        entries = os.listdir(_BY_ID_DIR)
    except OSError:
        entries = []
    for name in sorted(entries):
        if any(hint in name for hint in hints):
            return os.path.realpath(os.path.join(_BY_ID_DIR, name))
    if fallback and os.path.exists(fallback):
        return fallback
    if not allow_any_acm:
        hint_text = " or ".join(hints)
        raise RuntimeError(
            f"no serial device matching {hint_text!r} in {_BY_ID_DIR}; "
            "check the GPS USB connection or set FOLLOWRTK_GPS_PORT"
        )
    # Last resort for non-GPS devices: any ttyACM at all.
    acm = sorted(glob.glob("/dev/ttyACM*"))
    return acm[0] if acm else fallback


def find_gps_port() -> str:
    override = os.getenv("FOLLOWRTK_GPS_PORT")
    if override:
        return override
    if os.path.exists(GPS_BRIDGE):
        return GPS_BRIDGE
    return _resolve_by_id(GPS_ID_HINTS, GPS_FALLBACK, allow_any_acm=False)


def find_odrive_port() -> str:
    return _resolve_by_id(ODRIVE_ID_HINTS, ODRIVE_FALLBACK)


def find_arduino_port() -> str:
    return _resolve_by_id(ARDUINO_ID_HINTS, ARDUINO_FALLBACK)
