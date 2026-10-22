/*
 * Cart FSD — Pedal Control Firmware (gas + brake, Arduino Mega 2560)
 *
 * Closed-loop-ish (bang-bang with deadband) position control of both the
 * gas and brake linear actuators, plus a host-heartbeat watchdog that
 * drops the gas and applies full brake if the USB link to the Jetson /
 * laptop drops, and a software e-stop input on pin 2.
 *
 * WHY THIS EXISTS
 *   The Mega is powered from the cart's 5 V buck (off the 48 V pack), not
 *   from USB. If the USB cable to the host is unplugged — or the host
 *   crashes / hangs — the Mega keeps running and can keep driving the
 *   actuators at whatever target it last received. This sketch prevents
 *   that: any gap >= HEARTBEAT_TIMEOUT_MS in host traffic trips FAILSAFE,
 *   which drives the gas to its MIN pot position and the brake to its MAX
 *   and holds them there until valid commands resume.
 *
 * Wiring (see docs/linear_actuators.md for the full map):
 *   Gas   BTS7960:  R_EN=12, R_PWM=10, L_EN=13, L_PWM=11, pot=A4
 *   Brake BTS7960:  R_EN= 6, R_PWM= 4, L_EN= 7, L_PWM= 5, pot=A0
 *   L_PWM drives the pedal DOWN (engage). R_PWM retracts (release).
 *
 * Host → Mega protocol (USB serial, 115200 8N1, newline-terminated):
 *   G <value>    set gas target,   value in [0.0 .. GAS_POT_MAX]
 *   B <value>    set brake target, value in [0.0 .. BRAKE_POT_MAX]
 *   S            stop: both targets = MIN (pedals released, still armed)
 *   H            heartbeat-only ping — no target change
 *   D            graceful disarm: release both pedals, park the watchdog
 *   Any byte received also counts as an implicit heartbeat.
 *
 * Mega → Host telemetry:
 *   INFO,<text>                        boot / arming messages
 *   STAT,g=<pot>,b=<pot>,tg=<tgt>,tb=<tgt>,hb=<age_ms>,fs=<0|1>,es=<0|1>
 *                                      sent every STATUS_INTERVAL_MS
 *   EVT,ESTOP,<0|1>                    software e-stop pressed / released
 *   ERR,<text>                         parse errors, limit hits, etc.
 */

// ``sketches/common`` is added to the include path by hardware/arduino-sketches/upload.py
// (and should be by any other build system that uses this sketch).
#include "cart_limits.h"

// ---- Pin map (see docs/linear_actuators.md) -----------------------------
const int GAS_R_EN   = 12;
const int GAS_R_PWM  = 10;
const int GAS_L_EN   = 13;
const int GAS_L_PWM  = 11;
const int GAS_POT    = A4;

const int BRAKE_R_EN  = 6;
const int BRAKE_R_PWM = 4;
const int BRAKE_L_EN  = 7;
const int BRAKE_L_PWM = 5;
const int BRAKE_POT   = A0;

// Software E-stop (see docs/estop.md). NC button, INPUT_PULLUP. With NC
// contacts: LOW = normal, HIGH = pressed OR wire broken (fail-safe).
const int ESTOP_PIN = 2;
const unsigned long ESTOP_DEBOUNCE_MS = 50UL;

// ---- Control tuning -----------------------------------------------------
// PWM duty per pedal while moving (0..255). Pulled from cart_limits.h so
// limits.py stays the single source of truth.

// Deadband around the target (same units as the pot: 0..1 normalized).
// If |pot - target| < DEADBAND the actuator is stopped, so we don't
// chatter forward/backward across the target.
const float DEADBAND = 0.015f;

// ---- Heartbeat watchdog -------------------------------------------------
// If no byte has been received from the host for this long, drop into
// FAILSAFE and retract both pedals to their MIN pot position. 300 ms is
// ~15x the 50 Hz host control period — tolerant of a stutter, tight
// enough that a human can't react faster than the cart will stop.
const unsigned long HEARTBEAT_TIMEOUT_MS = 300UL;

// Status line cadence. Rate-limited so we don't flood the USB pipe.
const unsigned long STATUS_INTERVAL_MS = 100UL;

// ---- Runtime state ------------------------------------------------------
enum MotorDir { DIR_STOP = 0, DIR_FORWARD = 1, DIR_BACKWARD = -1 };

struct Pedal {
  const char *name;
  // Pin config
  int r_en, r_pwm, l_en, l_pwm, pot_pin;
  float pot_min, pot_max;
  uint8_t pwm_speed;
  // State
  float target;       // where we want to be (normalized pot units)
  float pot;          // most-recent pot reading
  MotorDir dir;
};

Pedal gas   = { "gas",   GAS_R_EN,   GAS_R_PWM,   GAS_L_EN,   GAS_L_PWM,   GAS_POT,
                GAS_POT_MIN,   GAS_POT_MAX,   GAS_ACTUATOR_PWM,
                GAS_POT_MIN,   0.0f, DIR_STOP };
Pedal brake = { "brake", BRAKE_R_EN, BRAKE_R_PWM, BRAKE_L_EN, BRAKE_L_PWM, BRAKE_POT,
                BRAKE_POT_MIN, BRAKE_POT_MAX, BRAKE_ACTUATOR_PWM,
                BRAKE_POT_MIN, 0.0f, DIR_STOP };

unsigned long last_host_byte_ms = 0;
unsigned long last_status_ms    = 0;
bool failsafe = true;   // start in failsafe; require a command to arm

// ---- E-stop state -------------------------------------------------------
// ``estop_active`` is the debounced logical state that drives pedal
// override (force brake full on, gas fully released). The ISR just sets
// a flag; the main loop does the heavy lifting so we never spend any
// serious time in interrupt context.
volatile bool estop_isr_event = false;
volatile unsigned long estop_last_isr_ms = 0;
bool estop_active = false;              // current debounced state
bool estop_last_pin_state = false;      // last raw read (LOW=normal, HIGH=estop)

void estopISR() {
  unsigned long now = millis();
  if (now - estop_last_isr_ms > ESTOP_DEBOUNCE_MS) {
    estop_isr_event = true;
    estop_last_isr_ms = now;
  }
}

// Line-buffered command parser.
char line_buf[32];
size_t line_len = 0;

// ---- Motor helpers ------------------------------------------------------
void driveStop(Pedal &p) {
  analogWrite(p.r_pwm, 0);
  analogWrite(p.l_pwm, 0);
  p.dir = DIR_STOP;
}

void driveForward(Pedal &p) {   // press pedal (engage)
  analogWrite(p.r_pwm, 0);
  analogWrite(p.l_pwm, p.pwm_speed);
  p.dir = DIR_FORWARD;
}

void driveBackward(Pedal &p) {  // release pedal (retract)
  analogWrite(p.l_pwm, 0);
  analogWrite(p.r_pwm, p.pwm_speed);
  p.dir = DIR_BACKWARD;
}

// Read pot, update p.pot, and return the 0..1 normalized value.
float readPot(Pedal &p) {
  int raw = analogRead(p.pot_pin);
  p.pot = (raw * (5.0f / 1023.0f)) / 5.0f;
  return p.pot;
}

// Core bang-bang step: move toward p.target, but never past p.pot_min /
// p.pot_max regardless of what the target says.
void stepPedal(Pedal &p) {
  float pot = readPot(p);
  // Clamp target to the mechanical envelope as a final safety net in case
  // the parser ever hands us a bogus value.
  float tgt = p.target;
  if (tgt < p.pot_min) tgt = p.pot_min;
  if (tgt > p.pot_max) tgt = p.pot_max;

  float err = tgt - pot;

  // Hard-stop at mechanical limits (belt-and-suspenders with the deadband).
  if (p.dir == DIR_FORWARD  && pot >= p.pot_max) { driveStop(p); return; }
  if (p.dir == DIR_BACKWARD && pot <= p.pot_min) { driveStop(p); return; }

  if (err > DEADBAND && pot < p.pot_max) {
    driveForward(p);
  } else if (err < -DEADBAND && pot > p.pot_min) {
    driveBackward(p);
  } else {
    driveStop(p);
  }
}

// ---- Command parsing ----------------------------------------------------
// Accepts:  G <float>  |  B <float>  |  S  |  H  |  D
void handleCommand(char *cmd) {
  switch (cmd[0]) {
    case 'G': {
      float v = atof(cmd + 1);
      if (v < 0.0f) v = 0.0f;
      if (v > GAS_POT_MAX + 1e-4f) { Serial.print(F("ERR,gas over cap: ")); Serial.println(v, 3); }
      if (v > GAS_POT_MAX) v = GAS_POT_MAX;
      gas.target = v;
      break;
    }
    case 'B': {
      float v = atof(cmd + 1);
      if (v < 0.0f) v = 0.0f;
      if (v > BRAKE_POT_MAX + 1e-4f) { Serial.print(F("ERR,brake over cap: ")); Serial.println(v, 3); }
      if (v > BRAKE_POT_MAX) v = BRAKE_POT_MAX;
      brake.target = v;
      break;
    }
    case 'S':
      gas.target   = gas.pot_min;
      brake.target = brake.pot_min;
      break;
    case 'H':
      // heartbeat only — timestamp already updated on the byte-read path
      break;
    case 'D':
      // Graceful disarm: host is shutting down on purpose. Release both
      // pedals AND set failsafe=true so the heartbeat watchdog does NOT
      // re-engage the brake once the host goes quiet. Returns early to
      // skip the post-switch failsafe-clear; otherwise we'd just go
      // straight back into the "host is alive" state and the watchdog
      // would slam the brake 300 ms later. Distinct from `S` (which
      // releases pedals but keeps the cart armed for more commands).
      gas.target   = gas.pot_min;
      brake.target = brake.pot_min;
      failsafe     = true;
      Serial.println(F("INFO,disarmed by host (graceful shutdown — pedals released, watchdog parked)"));
      return;
    default:
      Serial.print(F("ERR,unknown cmd: "));
      Serial.println(cmd);
      return;
  }

  // Any valid command clears FAILSAFE. The watchdog will re-trip it the
  // moment the host goes quiet again.
  if (failsafe) {
    failsafe = false;
    Serial.println(F("INFO,armed: heartbeat received, leaving failsafe"));
  }
}

void pumpSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    last_host_byte_ms = millis();

    if (c == '\n' || c == '\r') {
      if (line_len > 0) {
        line_buf[line_len] = '\0';
        handleCommand(line_buf);
        line_len = 0;
      }
    } else if (line_len < sizeof(line_buf) - 1) {
      line_buf[line_len++] = c;
    } else {
      // Overflowed a single line — drop it and complain. Normal commands
      // are way shorter than the buffer, so this is almost always noise.
      line_len = 0;
      Serial.println(F("ERR,line overflow — buffer reset"));
    }
  }
}

void checkHeartbeat() {
  unsigned long age = millis() - last_host_byte_ms;
  if (!failsafe && age > HEARTBEAT_TIMEOUT_MS) {
    failsafe = true;
    // Real failsafe: host disappeared → slam the brake, drop the gas.
    // Releasing both (the old behavior) made the cart coast, which is
    // the wrong answer when the operator's controller just dropped.
    gas.target   = gas.pot_min;
    brake.target = brake.pot_max;
    Serial.print(F("ERR,heartbeat timeout after "));
    Serial.print(age);
    Serial.println(F(" ms — FAILSAFE engaged (brake=MAX, gas=0)"));
  }
}

// Poll the e-stop pin, debounce, and on a state change:
//   - emit EVT,ESTOP,1|0 for the host to act on (idle ODrive, stop loop)
//   - while pressed, force brake to MAX and gas to MIN so the cart brakes
//     even if the host is missing in action. The override is applied here,
//     AFTER pumpSerial() handled any inbound G/B, so it always wins.
void checkEstop() {
  // Polled read drives edge detection for both press and release; the
  // ISR only flags a press so we don't miss it when loop() is slow.
  bool raw = digitalRead(ESTOP_PIN) == HIGH;
  if (raw != estop_last_pin_state) {
    // Debounce via the same window the ISR uses.
    unsigned long now = millis();
    if (now - estop_last_isr_ms > ESTOP_DEBOUNCE_MS) {
      estop_last_pin_state = raw;
      if (raw != estop_active) {
        estop_active = raw;
        Serial.print(F("EVT,ESTOP,")); Serial.println(estop_active ? 1 : 0);
        if (estop_active) {
          Serial.println(F("INFO,ESTOP engaged — full brake, gas released"));
        } else {
          Serial.println(F("INFO,ESTOP released — accepting commands again"));
        }
      }
      estop_last_isr_ms = now;
    }
  }
  // Consume any ISR-flagged press the poll missed (blocked loop iteration).
  if (estop_isr_event) {
    noInterrupts(); estop_isr_event = false; interrupts();
    if (!estop_active) {
      estop_active = true;
      estop_last_pin_state = true;
      Serial.println(F("EVT,ESTOP,1"));
      Serial.println(F("INFO,ESTOP engaged (via ISR) — full brake, gas released"));
    }
  }

  // Whenever the e-stop is active, slam targets regardless of what the
  // host last asked for. Re-applied every loop so even a rogue G/B
  // command can't escape the override for a single cycle.
  if (estop_active) {
    gas.target   = gas.pot_min;
    brake.target = brake.pot_max;
  }
}

void emitStatus() {
  unsigned long now = millis();
  if (now - last_status_ms < STATUS_INTERVAL_MS) return;
  last_status_ms = now;

  unsigned long age = now - last_host_byte_ms;
  Serial.print(F("STAT,g="));   Serial.print(gas.pot, 3);
  Serial.print(F(",b="));       Serial.print(brake.pot, 3);
  Serial.print(F(",tg="));      Serial.print(gas.target, 3);
  Serial.print(F(",tb="));      Serial.print(brake.target, 3);
  Serial.print(F(",hb="));      Serial.print(age);
  Serial.print(F(",fs="));      Serial.print(failsafe ? 1 : 0);
  Serial.print(F(",es="));      Serial.println(estop_active ? 1 : 0);
}

// ---- Setup / loop -------------------------------------------------------
void setup() {
  Serial.begin(115200);

  // The AVR toolchain's libstdc++ doesn't ship <initializer_list>, so we
  // can't range-for over a brace list here — use a plain const array.
  static const int output_pins[] = {
      GAS_R_EN, GAS_R_PWM, GAS_L_EN, GAS_L_PWM,
      BRAKE_R_EN, BRAKE_R_PWM, BRAKE_L_EN, BRAKE_L_PWM,
  };
  for (size_t i = 0; i < sizeof(output_pins) / sizeof(output_pins[0]); ++i) {
    pinMode(output_pins[i], OUTPUT);
  }
  // BTS7960 enables are tied high so the bridges respect PWM.
  digitalWrite(GAS_R_EN, HIGH);
  digitalWrite(GAS_L_EN, HIGH);
  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  driveStop(gas);
  driveStop(brake);

  // E-stop input + interrupt. NC contacts + INPUT_PULLUP: LOW = normal,
  // HIGH = pressed or wire broken. RISING catches the press; the polled
  // check in checkEstop() catches both edges for observability.
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  bool initial_estop = digitalRead(ESTOP_PIN) == HIGH;
  estop_active = initial_estop;
  estop_last_pin_state = initial_estop;
  attachInterrupt(digitalPinToInterrupt(ESTOP_PIN), estopISR, RISING);

  // Prime the heartbeat to "already stale" so we come up in FAILSAFE and
  // only leave it once the host proves it's alive.
  last_host_byte_ms = millis() - HEARTBEAT_TIMEOUT_MS - 1;

  Serial.println(F("INFO,Cart FSD pedal_control starting"));
  Serial.println(F("INFO,protocol: G<f>\\n | B<f>\\n | S\\n | H\\n  (any byte = heartbeat)"));
  Serial.print  (F("INFO,heartbeat timeout = ")); Serial.print(HEARTBEAT_TIMEOUT_MS);
  Serial.println(F(" ms"));
  Serial.print  (F("INFO,e-stop pin = ")); Serial.print(ESTOP_PIN);
  Serial.print  (F(" (initial state: ")); Serial.print(initial_estop ? F("PRESSED") : F("normal"));
  Serial.println(F(")"));
  if (initial_estop) {
    Serial.println(F("EVT,ESTOP,1"));
  }
  Serial.println(F("INFO,booting in FAILSAFE — send any command to arm"));
}

void loop() {
  pumpSerial();
  checkHeartbeat();
  checkEstop();   // must run AFTER pumpSerial so e-stop overrides commands

  stepPedal(gas);
  stepPedal(brake);

  emitStatus();
}
