# Cart FSD — Pedal Actuators (Gas & Brake)

The gas and brake pedals are each pressed by a 12 V linear actuator driven by a BTS7960 H-bridge, commanded by an Arduino Mega 2560. Position feedback comes from a linear potentiometer mechanically coupled to each actuator.

For steering, see [`steering.md`](./steering.md). For the e-stop button, see [`estop.md`](./estop.md).

---

## Hardware

| Component         | Part                               | Notes                                    |
|-------------------|------------------------------------|------------------------------------------|
| Controller        | Arduino Mega 2560                  | Also hosts the software e-stop (pin 2)   |
| Gas H-bridge      | BTS7960 (43 A module)              | Drives gas linear actuator               |
| Brake H-bridge    | BTS7960 (43 A module)              | Drives brake linear actuator             |
| Actuators         | 12 V linear actuators (×2)         | One per pedal                            |
| Feedback          | Linear potentiometers (×2)         | Mechanically coupled to each actuator    |

### Power

Both H-bridges run off a **12 V rail** derived from the cart's 48 V battery through a buck converter. Each actuator has its own fuse block, **5 A per actuator**. See [`steering.md`](./steering.md#power) for the full cart power map.

---

## Pin mapping

### Gas actuator (BTS7960)

| BTS7960 pin | Arduino Mega |
|-------------|--------------|
| R_EN        | 12           |
| R_PWM       | 10           |
| L_EN        | 13           |
| L_PWM       | 11           |

### Brake actuator (BTS7960)

| BTS7960 pin | Arduino Mega |
|-------------|--------------|
| R_EN        | 6            |
| R_PWM       | 4            |
| L_EN        | 7            |
| L_PWM       | 5            |

### Potentiometer inputs

| Pot       | Arduino pin |
|-----------|-------------|
| Gas pot   | A4          |
| Brake pot | A0          |

### Direction convention

- **L_PWM** = forward → pedal **engaged** (gas pressed, brake pressed).
- **R_PWM** = backward → pedal **released**.

> **Gotcha:** the original Notion spec had the gas and brake actuator pins *swapped*. The mapping above is the **corrected** wiring that matches the current cart and the sketches below. If you see pin numbers that look like they belong to the other pedal, check Notion first.

---

## Calibration — pot limits & safety stops

The pots output 0–5 V across their **full physical travel** (normalized 0.0 → 1.0). Each pedal only uses a **subset** of that travel:

- The actuator physically can't press the pedal past a certain point → defines the `MAX`.
- There's no reason to retract past the pedal's resting position → defines the `MIN`.

Both sketches enforce these limits in software: **when the pot reading hits a limit in the direction the motor is currently moving, the motor stops automatically.**

### Current calibrated limits

| Pedal | MIN  | MAX  | Meaning of MAX                                                        |
|-------|------|------|-----------------------------------------------------------------------|
| Gas   | 0.00 | 0.68 | Full throttle                                                         |
| Brake | 0.00 | 0.45 | Full brake (actuator can't press further, but brakes are fully engaged) |

These numbers live in code in two mirrored places — **edit both when calibration changes**:

- Python: [`limits.py`](../hardware/arduino-sketches/limits.py) — `GAS_POT_MIN/MAX`, `BRAKE_POT_MIN/MAX`
- Arduino: [`sketches/common/cart_limits.h`](../hardware/arduino-sketches/sketches/common/cart_limits.h) — same names, `float` constants

### How the limit logic works

- `f` (forward) drives the pedal down. If `normalized ≥ POT_MAX`, motor stops.
- `b` (backward) retracts the pedal. If `normalized ≤ POT_MIN`, motor stops.
- `s` always stops the motor immediately.
- A `direction` state variable tracks which way the motor is moving, so the limit check only blocks motion that would push *past* the limit — you can always reverse away from a limit.

If calibration drifts (new pedal, worn linkage, replaced pot), tweak `POT_MIN` / `POT_MAX` at the top of the relevant sketch.

---

## Usage

Both sketches expose the same serial interface at **115200 baud**:

| Command | Action   |
|---------|----------|
| `f`     | Forward (engage pedal)  |
| `b`     | Backward (release pedal)|
| `s`     | Stop immediately        |

While running, the sketch streams `voltage | normalized` on every loop so you can watch the pot move.

---

## Safety

- Each actuator has a **5 A fuse** on its 12 V feed.
- Software pot-limit stops prevent the actuator from driving past mechanical hard stops.
- **Hardware e-stop** (48 V battery cutoff) kills both actuators instantly along with everything else on the cart — see [`estop.md`](./estop.md).
- **Host-heartbeat watchdog** (the combined pedal firmware, below). The Mega is powered from the cart's 5 V buck, not from USB, so a yanked host cable does **not** power-cycle it — without a watchdog it would happily hold whatever throttle/brake target it last received. The `pedal_control.ino` sketch detects ~300 ms of silence from the host, retracts both pedals to their `POT_MIN`, and stays in FAILSAFE until valid commands resume. The Python side (`scripts/ps5_drive.py`) also detects `SerialException` on write and aborts the control loop so steering doesn't keep moving an uncommanded cart.
- The **software e-stop** button on Arduino pin 2 is handled by `pedal_control.ino` with a debounced ISR plus a polled check. While it's pressed, the brake target is forced to `BRAKE_POT_MAX` and the gas target to `GAS_POT_MIN`, overriding any `G`/`B` commands from the host. See [`estop.md`](./estop.md#firmware-behavior-pedal_controlino).

---

## Combined pedal firmware: `sketches/pedal_control/pedal_control.ino`

This is the sketch that actually runs on the cart — the `f`/`b`/`s` test sketches below are scaffolding for bench-bringup only. The combined firmware drives both actuators with pot-position feedback, accepts per-pedal targets from the host, and runs the heartbeat watchdog.

### Serial protocol (USB, 115200 8N1, newline-terminated)

**Host → Mega**

| Line        | Meaning                                                              |
|-------------|----------------------------------------------------------------------|
| `G <value>` | Set gas target pot value in `[0.0 .. GAS_POT_MAX]` (clamped).        |
| `B <value>` | Set brake target pot value in `[0.0 .. BRAKE_POT_MAX]` (clamped).    |
| `S`         | Stop: both targets = `POT_MIN` (same end-state as FAILSAFE).         |
| `H`         | Heartbeat-only ping — don't touch targets, just pet the watchdog.    |

Any received byte also counts as an implicit heartbeat — the 50 Hz `G <v>\nB <v>\n` stream from `ps5_drive.py` keeps the link healthy without needing separate `H` pings.

**Mega → Host**

| Prefix | Payload                                                                                                     |
|--------|-------------------------------------------------------------------------------------------------------------|
| `INFO,`| Boot / arming messages.                                                                                     |
| `STAT,`| `g=<pot>,b=<pot>,tg=<tgt>,tb=<tgt>,hb=<age_ms>,fs=<0|1>,es=<0|1>` — emitted every 100 ms. `fs=1` means FAILSAFE, `es=1` means the software e-stop is active. |
| `EVT,` | `ESTOP,<0|1>` — emitted when the software e-stop is pressed or released.                                     |
| `ERR,` | Parse errors, over-cap commands, line overflows, watchdog trips.                                            |

### Heartbeat watchdog

- `HEARTBEAT_TIMEOUT_MS = 300` (≈ 15× the 50 Hz host control period).
- **Boot state is FAILSAFE** — the sketch refuses to move either actuator until the host sends at least one valid command. This makes it safe to plug the Mega in before the Jetson is ready.
- Any gap ≥ `HEARTBEAT_TIMEOUT_MS` between received bytes → FAILSAFE: both targets set to `POT_MIN`, actuators driven backward until they retract, then held stopped. An `ERR,heartbeat timeout` line is emitted on the transition.
- Any valid command line clears FAILSAFE automatically (logs `INFO,armed`).

### Control behavior

- Bang-bang with a `DEADBAND = 0.015` (normalized pot units) around the target, so the actuator doesn't chatter forward/backward across its setpoint.
- Hard mechanical limits (`GAS_POT_MIN/MAX`, `BRAKE_POT_MIN/MAX` from `../common/cart_limits.h`) are enforced independently of the target — even a misbehaving host can't drive the pedals past their mechanical stops.

---

---

## Gas pedal test sketch

```cpp
const int GAS_R_EN  = 12;
const int GAS_R_PWM = 10;
const int GAS_L_EN  = 13;
const int GAS_L_PWM = 11;

const int GAS_POT = A4;

const int PWM_SPEED = 80;

// Normalized pot limits (0.0 = fully retracted, 1.0 = full pot travel)
const float POT_MIN = 0.00;
const float POT_MAX = 0.68;  // full throttle

// Direction state: 0 = stopped, 1 = forward, -1 = backward
int direction = 0;

void stopMotor() {
  analogWrite(GAS_R_PWM, 0);
  analogWrite(GAS_L_PWM, 0);
  direction = 0;
}

void driveForward() {
  analogWrite(GAS_R_PWM, 0);
  analogWrite(GAS_L_PWM, PWM_SPEED);
  direction = 1;
}

void driveBackward() {
  analogWrite(GAS_L_PWM, 0);
  analogWrite(GAS_R_PWM, PWM_SPEED);
  direction = -1;
}

void setup() {
  Serial.begin(115200);

  pinMode(GAS_R_EN, OUTPUT);
  pinMode(GAS_R_PWM, OUTPUT);
  pinMode(GAS_L_EN, OUTPUT);
  pinMode(GAS_L_PWM, OUTPUT);

  digitalWrite(GAS_R_EN, HIGH);
  digitalWrite(GAS_L_EN, HIGH);

  Serial.println("f = forward, b = backward, s = stop");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'f') {
      driveForward();
      Serial.println("Forward");
    } else if (c == 'b') {
      driveBackward();
      Serial.println("Backward");
    } else if (c == 's') {
      stopMotor();
      Serial.println("Stopped");
    }
  }

  float voltage = analogRead(GAS_POT) * (5.0 / 1023.0);
  float normalized = voltage / 5.0;

  // Safety limit: stop motor if past a limit in the direction of motion
  if (direction == 1 && normalized >= POT_MAX) {
    stopMotor();
    Serial.println("MAX limit reached - stopped");
  } else if (direction == -1 && normalized <= POT_MIN) {
    stopMotor();
    Serial.println("MIN limit reached - stopped");
  }

  Serial.print(voltage, 3);
  Serial.print("V | ");
  Serial.println(normalized, 3);
  delay(50);
}
```

---

## Brake pedal test sketch

```cpp
const int BRAKE_R_EN  = 6;
const int BRAKE_R_PWM = 4;
const int BRAKE_L_EN  = 7;
const int BRAKE_L_PWM = 5;

const int BRAKE_POT = A0;

const int PWM_SPEED = 80;

// Normalized pot limits (0.0 = fully retracted, 1.0 = full pot travel)
const float POT_MIN = 0.00;
const float POT_MAX = 0.45;  // full brake (mechanical limit)

// Direction state: 0 = stopped, 1 = forward, -1 = backward
int direction = 0;

void stopMotor() {
  analogWrite(BRAKE_R_PWM, 0);
  analogWrite(BRAKE_L_PWM, 0);
  direction = 0;
}

void driveForward() {
  analogWrite(BRAKE_R_PWM, 0);
  analogWrite(BRAKE_L_PWM, PWM_SPEED);
  direction = 1;
}

void driveBackward() {
  analogWrite(BRAKE_L_PWM, 0);
  analogWrite(BRAKE_R_PWM, PWM_SPEED);
  direction = -1;
}

void setup() {
  Serial.begin(115200);

  pinMode(BRAKE_R_EN, OUTPUT);
  pinMode(BRAKE_R_PWM, OUTPUT);
  pinMode(BRAKE_L_EN, OUTPUT);
  pinMode(BRAKE_L_PWM, OUTPUT);

  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  Serial.println("f = forward, b = backward, s = stop");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'f') {
      driveForward();
      Serial.println("Forward");
    } else if (c == 'b') {
      driveBackward();
      Serial.println("Backward");
    } else if (c == 's') {
      stopMotor();
      Serial.println("Stopped");
    }
  }

  float voltage = analogRead(BRAKE_POT) * (5.0 / 1023.0);
  float normalized = voltage / 5.0;

  // Safety limit: stop motor if past a limit in the direction of motion
  if (direction == 1 && normalized >= POT_MAX) {
    stopMotor();
    Serial.println("MAX limit reached - stopped");
  } else if (direction == -1 && normalized <= POT_MIN) {
    stopMotor();
    Serial.println("MIN limit reached - stopped");
  }

  Serial.print(voltage, 3);
  Serial.print("V | ");
  Serial.println(normalized, 3);
  delay(50);
}
```
