# arduino-sketches

Arduino Mega 2560 sketches that sit between the Jetson and the actuators.

| Path | What it does |
|------|--------------|
| `sketches/pedal_control/` | Gas + brake linear-actuator position control, pot feedback, host-heartbeat watchdog (FAILSAFE → brake), e-stop input. Speaks the `G/B/S/H/D` serial protocol used by `cartlib.pedals`. |
| `sketches/sensor_validation/` | GPS NMEA passthrough for bring-up. |
| `sketches/common/cart_limits.h` | C mirror of the limits in `limits.py`. |
| `limits.py` | Python source of truth for mechanical and software limits; `cartlib/config.py` imports it. |
| `upload.py` | Compile + flash with `arduino-cli`, with `sketches/common` on the include path. |

```bash
python hardware/arduino-sketches/upload.py                   # list sketches
python hardware/arduino-sketches/upload.py pedal_control     # build + flash
```

See [`docs/linear_actuators.md`](../../docs/linear_actuators.md) and
[`docs/estop.md`](../../docs/estop.md) for wiring and the safety model.
