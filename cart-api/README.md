# cart_api — Python control library for the FollowRTK golf cart

A small, organized Python library to **control** and **read** the self-driving
golf cart: RTK GPS, steering, gas, and brake — through one simple interface.

```python
from cartlib import Cart

with Cart() as cart:
    cart.arm()                       # GPS streaming; pedals armed (out of failsafe)
    print(cart.snapshot())           # live gps + pedals + steering state

    cart.steering.enable()           # energize steering motor
    cart.steering.set_angle(15)      # +15° at the steering column
    cart.pedals.set_brake(0.2)       # engage brake (safe)
    cart.stop()                      # release pedals
```

## Hardware map

| Subsystem | Device                         | Port (auto-detected by-id) | Protocol                          |
|-----------|--------------------------------|----------------------------|-----------------------------------|
| RTK GPS   | u-blox ZED-F9x GNSS            | `/dev/ttyACM0`             | NMEA (GGA) read; NTRIP for RTK    |
| Steering  | ODrive S1 + M8325s (3:1 belt)  | `/dev/ttyACM1`             | ODrive **ASCII** over USB-CDC     |
| Gas+Brake | Arduino Mega 2560 (2 actuators)| `/dev/ttyACM2`             | `pedal_control.ino` G/B/S/H/D     |

Ports are resolved from the stable `/dev/serial/by-id/` symlinks, so the
library keeps working even if the `ttyACM` numbers shuffle on reboot.

## Install

The only dependency is **pyserial** (already on the Jetson). No `odrive`
package needed — steering uses the ODrive ASCII protocol directly.

```bash
pip install -r requirements.txt   # pyserial>=3.5
```

## Layout

```
cart_api/
├── cartlib/                 the library
│   ├── config.py            ports, baud rates, limits (re-exported from limits.py)
│   ├── gps.py               GpsReceiver — read RTK GPS fixes
│   ├── ntrip.py             NtripClient — optional RTK corrections feeder
│   ├── pedals.py            PedalController — gas + brake (+ heartbeat watchdog)
│   ├── steering.py          SteeringController — ODrive S1 steering
│   └── cart.py              Cart — unifies all three
├── examples/
│   ├── read_all.py          live read-only dashboard
│   └── actuation_demo.py    opt-in brake / steer / gas demos (MOVES HARDWARE)
├── selftest.py              read-only verification of all 3 subsystems
└── requirements.txt
```

## Verify the hardware (no motion)

```bash
python3 selftest.py
```

Opens each device and reads live state — GPS fix, gas/brake pot positions,
e-stop/failsafe flags, steering angle, ODrive bus voltage. Exit code `0` =
everything detected and readable. Last run on the cart:

```
[GPS]      fix=GPS sats=12 hdop=0.54  lat=37.426562 lon=-122.164102      PASS
[PEDALS]   gas_pot=0.007 brake_pot=0.009  failsafe=True estop=False      PASS
[STEERING] bus_voltage=47.5 V  angle=-0.01°  state=IDLE  errors=0        PASS
```

## API reference

### `GpsReceiver` (`cartlib.gps`)
- `open()` / `close()` (or use as a context manager)
- `.latest` → `{lat, lon, fix_type, fix_code, sats, hdop, alt, ts}`
- `.wait_for_fix(timeout=10)`

### `NtripClient` (`cartlib.ntrip`)
- `NtripClient(gps).start()` / `.stop()` — feeds RTCM corrections to the
  receiver so it can reach **RTK Fixed**. **Note:** this opens an *outbound*
  connection to an NTRIP caster and sends the cart's position; it's opt-in.

### `PedalController` (`cartlib.pedals`)
- `arm()` — leave failsafe (starts the 20 Hz heartbeat keeping the Mega armed)
- `set_gas(v)` — `0..gas_cap` (**drives the cart**; default cap = `FSD_GAS_LIMIT` 0.25)
- `set_brake(v)` — `0..0.45` (always safe — only stops the cart)
- `stop()` — release both pedals · `disarm()` — graceful shutdown
- `.telemetry` → `{gas, brake, gas_target, brake_target, heartbeat_ms, failsafe, estop}`

### `SteeringController` (`cartlib.steering`)
- `enable()` — closed-loop, motor energized · `idle()` — de-energize
- `set_angle(deg)` — steering-column angle relative to connect() (clamped to ±90° default)
- `.angle_deg()`, `.bus_voltage()`, `.status()`, `clear_errors()`

### `Cart` (`cartlib.cart`)
- `arm()`, `stop()`, `emergency_brake()`, `snapshot()`
- `.gps`, `.pedals`, `.steering` — the subsystem objects

## Safety notes

- The Arduino boots in **FAILSAFE** and re-trips it if the host heartbeat
  stops for >300 ms (gas released, **brake slammed on**). `PedalController`
  runs the heartbeat for you while armed; closing it sends a graceful disarm.
- A hardware **e-stop** forces full brake / zero gas at the firmware level and
  surfaces as `telemetry["estop"]`.
- Gas is governed by a layered cap hierarchy (`config.effective_gas_cap`):
  hardware `GAS_POT_MAX` 0.68 → `GLOBAL_SPEED_LIMIT` 0.45 → mode cap.
- `examples/actuation_demo.py` moves real hardware and is fully opt-in; the
  `--gas` demo additionally requires `--i-understand-this-drives`.
- Limits are loaded by `config.py` from the shared
  [`hardware/arduino-sketches/limits.py`](../hardware/arduino-sketches/limits.py), which is mirrored in C by
  [`hardware/arduino-sketches/sketches/common/cart_limits.h`](../hardware/arduino-sketches/sketches/common/cart_limits.h). Edit those two files, not `config.py`.
```
