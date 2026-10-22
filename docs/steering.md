# Cart FSD — Steering System

The steering column is driven by a brushless motor through a belt reduction, closed-loop position-controlled by an ODrive S1. The Jetson AGX Thor talks to the ODrive over USB.

---

## Hardware

| Component              | Part                                   | Notes                                    |
|------------------------|----------------------------------------|------------------------------------------|
| Motor controller       | ODrive S1                              | USB-C to Jetson through a USB isolator   |
| Motor                  | M8325s-100KV BLDC                      | ~56 V max, runs directly off 48 V pack   |
| Encoder                | ODrive S1 on-board encoder             | Used for both commutation and position   |
| Reduction              | HTD 5M belt, 20T motor → 60T column    | **3:1** (3 motor turns = 1 wheel turn)   |
| Host                   | Jetson AGX Thor                        | USB-C → USB isolator → ODrive S1         |

### Mechanical convention

- 1 wheel turn = 360° at the steering column = **3 motor turns**.
- Positive motor position corresponds to positive steering-wheel degrees (sign convention set by how the belt is installed; confirm before every run).

---

## Power

Everything on the cart runs off the **48 V traction battery**. Voltage for each subsystem is derived from that pack:

| Rail       | Source                   | Feeds                          | Protection            |
|------------|--------------------------|--------------------------------|-----------------------|
| 48 V       | Battery (direct)         | ODrive S1 (steering motor)     | Fuse on main bus      |
| 12 V       | 48 V → 12 V buck         | Linear actuators (gas/brake)   | 5 A fuse per actuator |
| 5 V / logic| 48 V → 5 V buck          | Arduino Mega, sensors          | Fuse block            |

Notes:
- The M8325s is rated for up to ~56 V, so running it directly off 48 V nominal (fully charged pack can sit around 54 V) is within spec.
- There is no separate current limit set in software on the ODrive side yet — fusing is the current hard limit. Setting `motor.config.current_soft_max` / `current_hard_max` in ODrive config is a **TODO**.

---

## ODrive calibration & configuration

The ODrive has been calibrated (motor + encoder) and the steering system has been driven successfully under closed-loop control. **However, the calibrated values are not yet saved to ODrive NVM**, so a power cycle will lose them.

### To retrieve the current (live) calibration values

Connect to the ODrive and run `odrivetool` from the project root:

```bash
uv run odrivetool
```

Then inside the shell, dump the values you want to record:

```python
# Motor parameters (populated by calibration)
odrv0.axis0.config.motor.pole_pairs
odrv0.axis0.config.motor.torque_constant
odrv0.axis0.config.motor.phase_resistance
odrv0.axis0.config.motor.phase_inductance

# Current/velocity limits
odrv0.axis0.config.motor.current_soft_max
odrv0.axis0.config.motor.current_hard_max
odrv0.axis0.controller.config.vel_limit

# Encoder config (onboard)
odrv0.axis0.config.commutation_encoder
odrv0.axis0.config.load_encoder

# Control gains
odrv0.axis0.controller.config.pos_gain
odrv0.axis0.controller.config.vel_gain
odrv0.axis0.controller.config.vel_integrator_gain
```

### To back up the full config to a file

```bash
uv run odrivetool backup-config docs/odrive_steering_config.json
```

This writes every configured field to JSON and is the source of truth you want checked into the repo once the tune is finalized.

### To persist the config to the ODrive itself

```python
odrv0.save_configuration()   # writes to NVM, then reboots the ODrive
```

**Open TODO:** do a full `backup-config`, review the values, then `save_configuration()` so the cart survives a power cycle without needing to re-calibrate.

---

## Homing & zero reference

**Current behavior:** the script takes `pos_estimate` at startup as "0°" (straight ahead). That means **whatever position the wheels are in when `main.py` launches becomes center.**

Operational consequence: before running the script, the user must manually center the steering column. There is no limit switch, no index search, and no absolute-position recovery across power cycles.

**Open TODO:** define a real homing strategy. Candidates:
- Mechanical limit switch at one lock, home on startup.
- Index pulse / absolute encoder so center survives power cycles.
- User-confirmed visual center with a physical alignment mark.

---

## End stops & soft limits

**Current behavior:** the script sweeps ±90° at the steering column. There is no ODrive soft-limit configured and no mechanical limit-switch protection.

The soft angle limits and the belt ratio live in code in two mirrored places — **edit both when calibration changes**:

| Constant (Python / C)                          | Value | Meaning                                     |
|------------------------------------------------|-------|---------------------------------------------|
| `STEERING_MIN_DEG`                             | -320.0 | Left soft limit, at the steering column    |
| `STEERING_MAX_DEG`                             | +320.0 | Right soft limit, at the steering column   |
| `STEERING_BELT_RATIO`                          | 3.0   | Motor turns per column turn (HTD 5M, 20T→60T) |

- Python: [`limits.py`](../hardware/arduino-sketches/limits.py)
- Arduino: [`sketches/common/cart_limits.h`](../hardware/arduino-sketches/sketches/common/cart_limits.h)

`main.py` imports `STEERING_MIN_DEG` / `STEERING_MAX_DEG` / `STEERING_BELT_RATIO` and uses them as the sweep amplitude and gear reduction — don't hard-code those numbers in new scripts.

**Open TODO:** once the mechanical lock-to-lock range of the steering column is measured, tighten `STEERING_MIN_DEG` / `STEERING_MAX_DEG` to real values and also set matching ODrive position soft limits so the motor physically cannot drive the column past its mechanical stops.

---

## Usage

### Run the steering sweep test

```bash
uv run python main.py
```

### What the script does

1. Connects to any attached ODrive (`odrive.find_any`).
2. Clears any active errors.
3. Enters `CLOSED_LOOP_CONTROL` on `axis0`.
4. Configures `TRAP_TRAJ` input mode (trapezoidal trajectory planner).
5. Records the current position as the zero reference.
6. Runs 5 full sweeps of `0° → +90° → 0° → −90° → 0°` at the wheel.
7. Each sweep ramps up: sweep *n* uses `n/5` of the max velocity, accel, and decel. This is a **test-only ease-in**, not the long-term behavior — normal autonomous operation will just command positions at the full configured limits.
8. Holds at 0° for 10 s, then disarms (`IDLE` state) on exit.

Trajectory limits (motor-side units, from `main.py`):

| Parameter          | Final-sweep value  | Units      |
|--------------------|--------------------|------------|
| `TRAP_VEL_MAX`     | 8.0                | turns/s    |
| `TRAP_ACCEL_MAX`   | 15.0               | turns/s²   |
| `TRAP_DECEL_MAX`   | 15.0               | turns/s²   |
| `MAX_ANGLE_DEG`    | 90.0               | deg @ wheel|
| `NUM_SWEEPS`       | 5                  | —          |

Converted to the wheel (÷ belt ratio of 3): final cruise speed is ≈ 2.67 wheel-turns/s, which is fast — the ease-in ramp is there so the first few sweeps are gentle while you watch for mechanical issues.

### Ctrl-C behavior

The `finally` block commands `input_pos = start_pos`, waits 2 s, switches `input_mode` back to `PASSTHROUGH`, and puts the axis in `IDLE`. The motor is not actively held after that — the column is free to move.

---

## Safety

Two independent e-stop systems exist on the cart. See [`estop.md`](./estop.md) for the Arduino-side button wiring and ISR.

1. **Hardware E-stop (primary).** A physical e-stop button cuts power from the 48 V battery to the entire cart. This kills the ODrive instantly — the steering motor goes limp. Use this for any real emergency.
2. **Software E-stop (secondary).** An NC button wired to Arduino Mega pin 2 (`INT0`) that the controller firmware can react to. The intended use is a passenger-accessible "please stop" button that triggers a graceful stop (brake actuator forward, gas actuator retract, steering → IDLE). **Exact reaction is still TBD** — see `estop.md`.
3. **Ctrl-C / script exit.** The script's `finally` disarms the axis (`IDLE`). This is not a safety mechanism, just a clean shutdown.

**Open TODO:** wire the software e-stop into a supervisor on the Jetson that can command the ODrive to `IDLE` over USB when the Arduino reports an e-stop event.

---

## Open questions / TODOs

- [ ] `save_configuration()` on the ODrive so calibration survives power cycles.
- [ ] Commit `docs/odrive_steering_config.json` (`odrivetool backup-config`).
- [ ] Measure mechanical lock-to-lock range of the steering column.
- [ ] Configure ODrive position soft limits based on that range.
- [ ] Set `current_soft_max` / `current_hard_max` in ODrive config (don't rely on the fuse alone).
- [ ] Decide on a real homing strategy (limit switch vs. absolute encoder vs. operator-confirmed center).
- [ ] Define software-e-stop reaction (how Jetson should command the ODrive when the Arduino reports an e-stop).
- [ ] Confirm sign convention: `+° at the wheel` → which direction do the road wheels physically turn?
