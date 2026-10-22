# Cart FSD — PS5 Drive

Human-driven control of the cart using a **PS5 DualSense** controller, over Bluetooth or USB-C. The script lives at [`scripts/ps5_drive.py`](https://github.com/ethandgoodhart/PRODUCTION/blob/main/scripts/ps5_drive.py) and is the reference "non-autonomy driver" of the cart — the autonomy stack will eventually produce the same commands from the same limits, just with a different input source.

For the layered gas-authority hierarchy referenced below, see [`architecture.md`](./architecture.md#cross-cutting-gas-authority-hierarchy). For the host-heartbeat contract with the Arduino, see [`linear_actuators.md`](./linear_actuators.md#combined-pedal-firmware-sketchespedal_controlpedal_controlino).

---

## Input mapping

| DualSense input | Cart output                                                                    |
|-----------------|--------------------------------------------------------------------------------|
| Left stick X    | Steering column angle (ODrive S1 position control, TRAP_TRAJ planner)          |
| R2 trigger      | Gas pedal target pot value, scaled by `effective_gas_cap(PS5_GAS_LIMIT)`       |
| L2 trigger      | Brake pedal target pot value, scaled by `BRAKE_POT_MAX`                        |
| Esc / Q / close window | Graceful stop (S to pedals, ODrive → IDLE)                              |

Deadzones and per-trigger calibration match [`scripts/ps5_controller_test.py`](https://github.com/ethandgoodhart/PRODUCTION/blob/main/scripts/ps5_controller_test.py) — change them there first, then copy here.

---

## Modes (`--mode`)

The script ships with a flag so you can bring up one subsystem at a time:

| Mode       | Opens ODrive | Opens Mega serial | Use case                                  |
|------------|--------------|-------------------|-------------------------------------------|
| `full`     | yes          | yes               | Default — drive the whole cart            |
| `steering` | yes          | no                | Wheel-only bench test, no pedal firmware  |
| `pedals`   | no           | yes               | Pedal-only bench test, no ODrive attached |

`--dry-run` skips both, so you can check the input mapping and UI without any hardware present.

---

## Limits applied

On every frame, before anything is sent to hardware:

```text
steer_deg = lx * PS5_STEERING_MAX_DEG
gas       = r2 * effective_gas_cap(PS5_GAS_LIMIT)
brake     = l2 * BRAKE_POT_MAX
```

Where:

- `PS5_STEERING_MAX_DEG = min(60°, STEERING_MAX_DEG, -STEERING_MIN_DEG)` — narrower than the cart-wide soft limit while we're still early in testing.
- `effective_gas_cap(PS5_GAS_LIMIT) = min(GAS_POT_MAX, GLOBAL_SPEED_LIMIT, PS5_GAS_LIMIT)` — pulling `GLOBAL_SPEED_LIMIT` down in [`limits.py`](../hardware/arduino-sketches/limits.py) always wins.

`PedalLink` clamps one more time against `GAS_POT_MAX` / `BRAKE_POT_MAX` before the write as a last-ditch safety net.

---

## Safety behaviors built in

- **Pedals first, steering after.** Each frame sends pedal targets before touching the ODrive; if the pedal write fails, steering is skipped that frame.
- **Serial fault aborts the loop.** `PedalLink` catches `OSError` / `SerialException` on every write, marks itself `faulted`, and the main loop breaks. The `finally` block then idles the ODrive and best-effort-sends `S` to whatever port is left.
- **Heartbeat UI.** The readout shows `heartbeat OK / slow / STALE` with ms-since-last-successful-write. `STALE` means the firmware is about to fail safe on its own (`HEARTBEAT_TIMEOUT_S = 0.30`, matched to the firmware's 300 ms).
- **Controller disconnect** (`JOYDEVICEREMOVED`) exits the loop cleanly.

Combined effect: **yanking the USB cable stops the cart.** Host sees the write error (one frame), Arduino stops seeing bytes, and the firmware watchdog releases the gas and applies full brake ~300 ms later regardless of what the host does.

---

## Usage

```bash
uv run python scripts/ps5_drive.py                 # full control (default)
uv run python scripts/ps5_drive.py --mode steering # steering only
uv run python scripts/ps5_drive.py --mode pedals   # pedals only
uv run python scripts/ps5_drive.py --dry-run       # no hardware at all
uv run python scripts/ps5_drive.py --arduino-port /dev/tty.usbmodem1401
```

The pygame window must be focused for input events to arrive.

---

## Pairing the DualSense (macOS)

1. System Settings → Bluetooth.
2. Hold **PS + Create** on the controller until the lightbar double-flashes.
3. Pair from the Bluetooth panel.
4. Confirm with `uv run python scripts/ps5_controller_test.py --list`.

USB-C over the data cable also works and is preferred for testing since there's no BT latency/jitter to worry about.

---

## Open TODOs

- [ ] Move `PedalLink` / `SteeringLink` into the `cart.pedals` and `cart.steering` packages so the FSD script can share them.
- [ ] Add a dedicated button binding for "arm" and "graceful stop" (Options + Circle) instead of relying on keyboard focus.
- [ ] Surface the Arduino's `STAT,…` telemetry in the UI (actual pot reading vs. target) once the Mega is sending it.
