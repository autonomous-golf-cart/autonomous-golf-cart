"""
Cart FSD — single source of truth for mechanical/software limits AND for
the layered per-mode caps on how much throttle a given controller is
allowed to command.

Two kinds of limit live here:

1. Hardware-side limits (pot travel, steering lock-to-lock). Driven by
   physics/mechanics — must never be violated regardless of who's
   driving. These stay mirrored in the matching C header for Arduino
   sketches, ``sketches/common/cart_limits.h`` — keep the two in sync.

2. A layered hierarchy of gas caps, by who's driving:

       GAS_POT_MAX          (hardware max — never exceed)
         ↑ always clamps
       GLOBAL_SPEED_LIMIT   (top-level, applies in EVERY mode)
         ↑ always clamps
       PS5_GAS_LIMIT        (human-driving-with-PS5-controller cap)
       FSD_GAS_LIMIT        (autonomy cap — lowest layer / most conservative)

   "Global has hierarchy over PS5" means: even if the PS5 script is told
   it can push gas to 0.9, the global cap will shrink that to
   GLOBAL_SPEED_LIMIT. "FSD sits at the lowest level" means the autonomy
   stack's own cap lives underneath everything else — it is the smallest
   number by default so robot-driving is the slowest authority tier on
   the cart.

Use ``effective_gas_cap(mode_limit)`` to resolve the cap for a given
mode — it takes ``min()`` across the hardware, global, and mode-specific
ceilings so callers can't accidentally bypass a layer above them.
"""

# --- Pedal linear actuators (normalized pot reading, 0.0 = fully retracted,
# 1.0 = full pot travel). See docs/linear_actuators.md. ---

# Gas pedal (actuator + linear pot on A4)
GAS_POT_MIN = 0.00   # resting / fully released
GAS_POT_MAX = 0.68   # full throttle (mechanical hard stop of the actuator)

# Brake pedal (actuator + linear pot on A0)
BRAKE_POT_MIN = 0.00  # resting / fully released
BRAKE_POT_MAX = 0.45  # full brake (mechanical limit of the actuator vs. pedal)

# Per-actuator PWM duty (0..255) while moving. Higher = faster actuator
# travel. The BTS7960 is a full H-bridge and the actuators are 12V-rated,
# so 255 = full rail is safe; pick a lower value if you want softer motion.
GAS_ACTUATOR_PWM = 255
BRAKE_ACTUATOR_PWM = 255


# --- Gas authority hierarchy ------------------------------------------------
#
# All values are in the same units as GAS_POT_MAX — a normalized pot reading
# where 0.0 = pedal fully released and GAS_POT_MAX = pedal fully engaged.
# Each layer must be <= the layer above it; the resolver below clamps with
# ``min()`` so a bug in a lower layer can never escape its container.

# Top-level governor. Applies regardless of who's driving (PS5 operator,
# FSD, bench test, anyone). Lower this number to slow the whole cart down
# in one place.
GLOBAL_SPEED_LIMIT = 0.45

# What full-squeeze R2 on the DualSense is allowed to command when the
# PS5 script is driving. Stays below GLOBAL_SPEED_LIMIT so that tightening
# the global knob always wins.
PS5_GAS_LIMIT = GLOBAL_SPEED_LIMIT

# Cap used by the autonomy stack. At the bottom of the hierarchy on
# purpose: the robot should drive slower than a human operator until it
# has earned more authority. 0.36 pot ≈ 12 mph, under the 0.45 governor.
FSD_GAS_LIMIT = 0.36


# --- Steering (ODrive S1 + M8325s through HTD 5M belt). See docs/steering.md. ---

# HTD 5M belt, 20T motor pulley -> 60T column pulley.
# 3 motor turns = 1 full turn of the steering column / steering wheel.
STEERING_BELT_RATIO = 3.0

# Soft angle limits measured AT THE STEERING COLUMN. Cart's rack
# physically supports ~±270° lock-to-lock; live we observed the wheel
# refusing to track past ±80° even though no mechanical stop was hit —
# almost certainly an ODrive vel_limit/current_limit choking the motor
# against the rising rack-load near full lock. Keeping the soft cap at
# the full mechanical range; see ps5_drive.py for the current/vel
# unblock attempts.
STEERING_MIN_DEG = -270.0
STEERING_MAX_DEG = 270.0


def steering_deg_to_motor_turns(column_deg: float) -> float:
    """Convert a steering-column angle (deg) to motor turns."""
    return (column_deg / 360.0) * STEERING_BELT_RATIO


def motor_turns_to_steering_deg(motor_turns: float) -> float:
    """Convert motor turns to a steering-column angle (deg)."""
    return (motor_turns / STEERING_BELT_RATIO) * 360.0


def effective_gas_cap(mode_limit: float) -> float:
    """
    Resolve the effective gas cap for a mode-specific limit.

    Always clamped down by the hardware ceiling (``GAS_POT_MAX``) and the
    project-wide governor (``GLOBAL_SPEED_LIMIT``). Callers pass their own
    cap (e.g. ``PS5_GAS_LIMIT`` or ``FSD_GAS_LIMIT``) and get back the
    value they're actually allowed to command — no layer above them can
    be bypassed by accident.
    """
    return min(GAS_POT_MAX, GLOBAL_SPEED_LIMIT, mode_limit)
