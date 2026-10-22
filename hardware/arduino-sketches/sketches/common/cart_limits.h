// Cart FSD — single source of truth for mechanical / software limits
// (Arduino side). Keep this in sync with hardware/arduino-sketches/limits.py and cart-api/cartlib/config.py.
//
// Include from any sketch that drives an actuator or the steering motor:
//   #include "../common/cart_limits.h"

#ifndef CART_LIMITS_H
#define CART_LIMITS_H

#include <stdint.h>

// ---- Pedal linear actuators (normalized pot reading 0.0..1.0) ---------------
// See docs/linear_actuators.md.

// Gas pedal (pot on A4)
static const float GAS_POT_MIN   = 0.00f;  // resting / fully released
static const float GAS_POT_MAX   = 0.68f;  // full throttle

// Brake pedal (pot on A0)
static const float BRAKE_POT_MIN = 0.00f;  // resting / fully released
static const float BRAKE_POT_MAX = 0.45f;  // full brake (mechanical limit)

// PWM duty (0..255) for each actuator while moving — higher = faster.
static const uint8_t GAS_ACTUATOR_PWM   = 230;
static const uint8_t BRAKE_ACTUATOR_PWM = 230;


// ---- Steering (ODrive S1 + M8325s through HTD 5M belt) ----------------------
// See docs/steering.md.

// HTD 5M belt, 20T motor -> 60T column. 3 motor turns = 1 column turn.
static const float STEERING_BELT_RATIO = 3.0f;

// Soft angle limits AT THE STEERING COLUMN (same frame as the steering wheel).
// The rack physically supports ~±320° lock-to-lock.
static const float STEERING_MIN_DEG = -320.0f;
static const float STEERING_MAX_DEG =  320.0f;

#endif  // CART_LIMITS_H
