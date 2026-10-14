# Cart FSD — Architecture

How the cart's software is organized today, where it's going, and the cross-cutting rules (limits hierarchy, safety layers) that every subsystem has to respect.

For per-subsystem detail see [`steering.md`](./steering.md), [`linear_actuators.md`](./linear_actuators.md), [`estop.md`](./estop.md), [`gps.md`](./gps.md), [`cameras.md`](./cameras.md), and [`ps5_drive.md`](./ps5_drive.md).

---

## Layers

From the road up:

| Layer            | Runs on              | Responsibilities                                                                 |
|------------------|----------------------|----------------------------------------------------------------------------------|
| Actuators        | Motors / H-bridges   | Physically move the cart (ODrive S1 steering, 2× BTS7960 linear actuators).      |
| Firmware         | Arduino Mega 2560    | Closed-loop-ish position control of both pedals, pot feedback, host heartbeat watchdog. GPS passthrough. |
| Host control     | Jetson AGX Thor (dev: MacBook) | Human + autonomy controllers. Talks USB to the ODrive, USB serial to the Mega, USB to 4 cameras. |
| Input sources    | Human or FSD         | PS5 DualSense (human) or a driving policy in `policies/` (FSD). Both produce the same commands.     |

Power is also layered — 48 V pack → dedicated 12 V and 5 V buck rails — see [`steering.md`](./steering.md#power).

---

## Repo shape

```text
autonomous-golf-cart/
├── cart-api/                      # cartlib: the one Python interface to the hardware
│   ├── cartlib/
│   │   ├── cart.py                # Cart: GPS + pedals + steering behind one handle
│   │   ├── steering.py            # ODrive S1 over USB-CDC ASCII
│   │   ├── pedals.py              # Mega serial protocol + heartbeat thread
│   │   ├── gps.py, ntrip.py       # u-blox RTK receiver + NTRIP corrections
│   │   ├── follow.py              # pure-pursuit path follower used by `cart drive`
│   │   ├── server.py              # WebSocket bridge for the RTK map UI
│   │   └── config.py              # ports + limits used at runtime
│   ├── cart                       # CLI: test / read / record / drive / serve
│   ├── examples/  tests/
├── policies/                      # driving policies; each one drives through cartlib
│   ├── rtk/                       # map UI, lane annotator, RTK waypoint following
│   └── segmentation/              # SegFormer → BEV → lane-aware planner
│       ├── segdrive/              # shared library
│       └── onboard/               # Jetson sidecar (segmentation_infer.py) + drive.py
├── hardware/
│   ├── arduino-sketches/
│   │   ├── limits.py              # limit values, see below
│   │   ├── upload.py              # arduino-cli compile + flash
│   │   └── sketches/
│   │       ├── common/cart_limits.h   # C mirror of limits.py
│   │       ├── pedal_control/         # gas + brake firmware w/ heartbeat watchdog
│   │       └── sensor_validation/     # GPS NMEA passthrough
│   ├── sensors/calibration/       # camera intrinsics/extrinsics, ChArUco boards
│   └── cad/  electrical/  mechanical/  pcb/
└── docs/
```

Every policy produces the same three commands (steering-column angle, gas pot,
brake pot) and sends them through `cartlib`. The human PS5 driver
([`ps5_drive.md`](./ps5_drive.md)) still lives in the separate
[PRODUCTION](https://github.com/ethandgoodhart/PRODUCTION) repo and talks to the
hardware directly.

---

## Cross-cutting: gas authority hierarchy

Every gas-pedal command, regardless of who's driving, is clamped by this stack (see [`limits.py`](../hardware/arduino-sketches/limits.py)):

```text
GAS_POT_MAX          hardware ceiling — the actuator physically can't go past this
  ↑ always clamps
GLOBAL_SPEED_LIMIT   project-wide governor — applies in every mode
  ↑ always clamps
PS5_GAS_LIMIT        human driving cap
FSD_GAS_LIMIT        autonomy cap (lowest layer — robot drives slower than a human by default)
```

`limits.effective_gas_cap(mode_limit)` returns `min(GAS_POT_MAX, GLOBAL_SPEED_LIMIT, mode_limit)`. Callers should never bypass this helper — lowering `GLOBAL_SPEED_LIMIT` must be sufficient to slow the whole cart down.

---

## Cross-cutting: safety layers

Listed in order of authority. Each lower layer assumes the ones above it may fail.

| Layer                 | Mechanism                                                                 | Failure mode caught                               |
|-----------------------|---------------------------------------------------------------------------|---------------------------------------------------|
| Hardware e-stop       | 48 V battery cutoff button                                                | Anything, including the Jetson catching fire      |
| Software e-stop (TBD) | NC button → Mega pin 2 INT0                                               | Passenger-reachable "please stop"                 |
| Firmware watchdog     | `pedal_control.ino` retracts both pedals if host silent for 300 ms         | USB cable yanked, host crashes, host hangs        |
| Firmware pot limits   | Hard stops at `*_POT_MIN/MAX` regardless of commanded target               | Misbehaving host, parser bugs                     |
| Host fault detection  | `PedalLink` aborts the loop on `SerialException`                          | Lost pedal link while still commanding steering   |
| Planner staleness     | `segmentation/onboard/drive.py` cuts gas at once, brakes after 1 s        | Planner sidecar crashes or hangs                  |
| Gas authority caps    | `effective_gas_cap()` — hardware / global / per-mode                      | Overzealous controller (human or robot)           |
| Steering soft limits  | `STEERING_MIN/MAX_DEG`, planner velocity/accel caps                        | Commanded angle outside mechanical lock-to-lock   |

The firmware watchdog and the host fault detection together mean: **yanking the USB cable stops the cart** even though the Mega keeps its own power. See [`linear_actuators.md`](./linear_actuators.md#combined-pedal-firmware-sketchespedal_controlpedal_controlino) for the protocol spec.

---

## Firmware workflow (no Arduino IDE)

`.ino` files are the Arduino IDE's native format — but they're really just C++ with a preprocessor that auto-generates prototypes and injects `#include <Arduino.h>`. We keep the extension (it's what every AVR tool expects) and drive the toolchain from the CLI via [`arduino-cli`](https://arduino.github.io/arduino-cli/), which is what the IDE's green arrow calls under the hood.

[`upload.py`](../hardware/arduino-sketches/upload.py) wraps `arduino-cli compile` + `arduino-cli upload` so the flow is one command:

```bash
python hardware/arduino-sketches/upload.py                               # show sketches
python hardware/arduino-sketches/upload.py pedal_control                 # compile + flash
python hardware/arduino-sketches/upload.py pedal_control --monitor       # then open serial
python hardware/arduino-sketches/upload.py pedal_control --compile-only  # just build
```

Mechanics:

- Board FQBN is `arduino:avr:mega` (works for Mega and Mega 2560).
- Sketches are auto-discovered as every `sketches/<name>/<name>.ino`.
- The Mega port auto-detects the same way the Python runtime scripts do — by USB VID, explicitly skipping the ODrive.
- Shared headers under `sketches/common/` are added to the compiler's `-I` search path, so any sketch can `#include "cart_limits.h"` without caring where the build dir ends up.
- `compile-only` is safe on a plugged-in Jetson — nothing is flashed unless upload is requested.
- One-time setup on a new machine: `brew install arduino-cli`; the script installs the `arduino:avr` core on demand.

---

## Cross-cutting: limits stay mirrored between Python and C

Three files hold the same limits:

- [`limits.py`](../hardware/arduino-sketches/limits.py): the reference Python copy.
- [`sketches/common/cart_limits.h`](../hardware/arduino-sketches/sketches/common/cart_limits.h): what every Arduino sketch `#include`s.
- [`cartlib/config.py`](../cart-api/cartlib/config.py): what the policies actually use at runtime.

When pot calibration or steering limits change, **edit all three** or one side will silently trust numbers the others have stopped believing.

> [!WARNING]
> They currently disagree. `FSD_GAS_LIMIT` is 0.36 in `cartlib/config.py` and
> 0.25 in `limits.py`. Steering limits are ±320° in `cartlib/config.py`, ±270°
> in `limits.py`, and ±90° in `cart_limits.h`. Actuator PWM is 255 in
> `limits.py` and 230 in `cart_limits.h`.

---

## Open TODOs (architecture-scope)

- [ ] Reconcile `cartlib/config.py` with `limits.py` / `cart_limits.h` (see the warning above), ideally by having `cartlib` import one shared copy.
- [ ] Extend `cart-api/tests/` with pytest coverage for pure-logic pieces (`effective_gas_cap`, steering conversions).
- [ ] Move `ps5_drive.py` onto `cartlib` and into this repo.
- [ ] Wire the software e-stop from Mega → Jetson so the ODrive gets idled when the Arduino reports an e-stop event (see [`estop.md`](./estop.md)).
- [ ] Stable physical-camera ↔ logical-name mapping so USB enumeration order stops mattering (see [`cameras.md`](./cameras.md)).
