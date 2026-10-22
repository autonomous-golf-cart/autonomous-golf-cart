"""
cartlib.follow — autonomous GPS path following for the cart.

A conservative cross-track controller that ties the RTK position to steering
and throttle WITHOUT ever computing a GPS heading:

  * Steering is a heading-free cross-track PD law. ``signed_distance_m`` from
    the path geometry tells us which side of the line we're on (and how far);
    its rate of change supplies the damping term that a heading would normally
    provide (cross_rate ~= v * sin(heading_error), so PD on
    (cross, cross_rate) is a Stanley-style law that needs no world heading).
    The cart's own steering-wheel angle sensor closes the inner loop.
  * Throttle ramps toward ``max_speed_mph`` using the live GPS speed as
    feedback, hard-capped at ``gas_cap`` and backed off in sharp turns.
  * Throttle is cut to zero (with brake) at the goal.

SAFETY — this AUTONOMOUSLY DRIVES THE CART. The follower:
  * caps gas at ``gas_cap`` (default = FSD_GAS_LIMIT, 0.36),
  * stops on e-stop, on lost/old GPS, on excessive cross-track error, and at
    the final waypoint,
  * defaults to DRY-RUN (computes + prints, sends nothing) unless armed.

Path format: a list of (lat, lon) tuples. ``load_path`` accepts JSON in the
drivelive ({lat,lng}), lane_tracker (fix dicts with lat/lon), or plain
[[lat,lon], ...] shapes.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import List, Optional

from . import config, geo
from .cart import Cart


def load_path(path_file: str) -> List[geo.LatLon]:
    """Load waypoints from a JSON file, tolerant of several shapes."""
    with open(path_file) as f:
        data = json.load(f)
    # Unwrap common containers.
    if isinstance(data, dict):
        for key in ("points", "path", "waypoints"):
            if key in data:
                data = data[key]
                break
    pts: List[geo.LatLon] = []
    for item in data:
        if isinstance(item, dict):
            lat = item.get("lat")
            lon = item.get("lon", item.get("lng"))
        else:  # [lat, lon] pair
            lat, lon = item[0], item[1]
        if lat is not None and lon is not None:
            pts.append((float(lat), float(lon)))
    if len(pts) < 2:
        raise ValueError(f"{path_file}: need >=2 waypoints, got {len(pts)}")
    return pts


@dataclass
class FollowConfig:
    lookahead_m: float = 2.0          # cross-track correction lookahead distance
    gas_cap: float = config.FSD_GAS_LIMIT   # hard ceiling the controller may push to
    max_speed_mph: float = 3.0        # target cruise speed (closed-loop setpoint)
    live_speed_mph: float = 0.0       # GPS-derived speed currently shown in UI
    # Closed-loop speed control: PI on the GPS speed (so we actually hit the
    # requested mph despite hills / load / open-loop miscalibration) plus a
    # brake term that bleeds overspeed on descents.
    speed_kp: float = 0.030           # gas pot added per mph of speed deficit
    speed_ki: float = 0.012           # gas pot per (mph*s) of accumulated error
    speed_i_max: float = 35.0         # clamp on the error integral (mph*s)
    brake_kp: float = 0.08            # brake pot per mph of overspeed
    brake_deadband_mph: float = 0.5   # tolerate this much overspeed before braking
    steer_gain: float = 1.3           # deg steering per deg cross-track correction
    xtrack_gain: float = 1.5          # multiplier on signed cross-track error
    heading_gain: float = 3.0         # D-gain: deg steer per deg of (cross-rate)
                                      # heading error — the damping that stops the
                                      # weave/overshoot. See step()'s heading_err.
    heading_min_speed_mph: float = 0.5  # below this, heading estimate is unreliable
    steer_sign: float = 1.0           # hardware steering sign convention
    max_steer_deg: float = 320.0      # clamp on commanded column angle
    turn_slowdown: float = 0.0        # gas *= 1/(1+turn_slowdown*|steer|/max_steer)
    goal_radius_m: float = 1.5        # within this of last point => arrived
    # Final-approach deceleration: ease the speed setpoint down to a crawl over
    # arrival_slowdown_m so we glide to the goal instead of cruising flat-out
    # until the cutoff and lurching, then hold a firm brake to a full stop.
    arrival_slowdown_m: float = 6.0   # start easing speed down within this of goal
    arrival_creep_mph: float = 1.0    # floor speed kept until inside goal_radius
    arrival_brake: float = 0.30       # brake pot held once arrived (full stop & park)
    max_crosstrack_m: float = 6.0     # abort if we stray this far off path
    # The Pi Zero GPS bridge delivers 10 Hz fixes but occasionally batches
    # them up to ~0.5 s late; keep the watchdog well clear of that jitter.
    gps_max_age_s: float = 1.5        # stop if usable GPS is absent this long
    rtk_loss_grace_s: float = 1.5     # stop if required RTK is absent this long
    require_rtk: bool = False         # require RTK Fixed/Float to drive
    rate_hz: float = 15.0


@dataclass
class FollowState:
    phase: str = "init"               # init | tracking | done | abort
    reason: str = ""


class PathFollower:
    def __init__(self, cart: Cart, path: List[geo.LatLon],
                 cfg: Optional[FollowConfig] = None, armed: bool = False):
        self.cart = cart
        self.path = path
        self.cfg = cfg or FollowConfig()
        self.cfg.gas_cap = config.effective_gas_cap(self.cfg.gas_cap)
        self.armed = armed             # False => dry-run (no actuator output)
        self.state = FollowState()
        self._last_steer_read_ts = 0.0
        self._last_actual_steer_deg: Optional[float] = None
        # cross-track derivative state (heading-free damping)
        self._prev_cross: Optional[float] = None
        self._prev_cross_ts: Optional[float] = None
        self._heading_ewma: Optional[float] = None   # smoothed wheel angle (needle)
        self._herr_ewma: Optional[float] = None       # smoothed cross-rate heading err
        # closed-loop speed control state
        self._speed_integral: float = 0.0
        self._prev_speed_ts: Optional[float] = None
        self._gps_loss_since: Optional[float] = None
        self._rtk_loss_since: Optional[float] = None

    # -- helpers -----------------------------------------------------------
    def _apply(self, gas: float, brake: float, steer_deg: Optional[float]) -> None:
        """Send (or, in dry-run, just record) actuator commands."""
        gas = max(0.0, min(gas, self.cfg.gas_cap))
        if not self.armed:
            return
        if steer_deg is not None and self.cart.steering and self.cart.steering._enabled:
            self.cart.steering.set_angle(steer_deg)
        if self.cart.pedals:
            self.cart.pedals.set_brake(brake)
            self.cart.pedals.set_gas(gas)

    def _stop(self, reason: str, brake: float = 0.15) -> None:
        self.state.reason = reason
        if self.armed and self.cart.pedals:
            self.cart.pedals.set_gas(0.0)
            self.cart.pedals.set_brake(brake)

    def _pause_for_signal(self, reason: str, fix=None) -> dict:
        """Keep the drive alive during a short GPS/RTK dropout, but command no
        throttle while we wait for the signal to recover or time out."""
        self.state.reason = reason
        self._apply(gas=0.0, brake=0.05, steer_deg=None)
        return self._telemetry(fix, None, 0.0, brake=0.05)

    def _cross_rate(self, cross: float, now: float) -> float:
        """d(signed cross-track)/dt — the heading-free damping signal."""
        rate = 0.0
        if self._prev_cross is not None and self._prev_cross_ts is not None:
            dt = now - self._prev_cross_ts
            if dt > 1e-3:
                rate = (cross - self._prev_cross) / dt
        self._prev_cross = cross
        self._prev_cross_ts = now
        return rate

    def _smooth_heading(self, wheel_deg: float) -> float:
        """Light EWMA on the measured wheel angle — just knocks off encoder
        quantization while still tracking the current angle almost exactly (the
        map needle leans on this 1:1). Render-side easing handles the rest."""
        if self._heading_ewma is None:
            self._heading_ewma = wheel_deg
        else:
            self._heading_ewma = 0.4 * self._heading_ewma + 0.6 * wheel_deg
        return self._heading_ewma

    def _smooth_heading_err(self, herr_deg: float) -> float:
        """Heavier EWMA on the cross-rate heading estimate — the numerical
        derivative of position is noisy, so this is the main thing keeping the
        damping term from chattering. Trades a little lag for a lot of calm."""
        if self._herr_ewma is None:
            self._herr_ewma = herr_deg
        else:
            self._herr_ewma = 0.6 * self._herr_ewma + 0.4 * herr_deg
        return self._herr_ewma

    # -- one control step --------------------------------------------------
    def step(self) -> dict:
        """Run one control cycle. Returns a telemetry dict for logging/UI."""
        c = self.cfg
        fix = self.cart.gps.latest if self.cart.gps else None
        ped = self.cart.pedals.telemetry if self.cart.pedals else {}

        # --- safety gates ---
        if ped.get("estop"):
            self.state.phase = "abort"
            self._stop("E-STOP engaged")
            return self._telemetry(fix, None, 0.0)
        now = time.time()
        gps_ok = bool(
            fix
            and fix.get("fix_code", 0) > 0
            and fix.get("lat") is not None
            and fix.get("lon") is not None
            and (now - fix["ts"]) <= c.gps_max_age_s
        )
        if not gps_ok:
            age = (now - fix["ts"]) if fix and "ts" in fix else None
            if self._gps_loss_since is None:
                self._gps_loss_since = now - max(age or 0.0, 0.0)
            lost_for = now - self._gps_loss_since
            if lost_for >= c.gps_max_age_s:
                self.state.phase = "abort"
                self._stop(f"GPS fix lost/stale for {lost_for:.2f}s")
                return self._telemetry(fix, None, 0.0)
            return self._pause_for_signal(f"GPS dropout {lost_for:.2f}s", fix)
        self._gps_loss_since = None

        if c.require_rtk and fix["fix_code"] not in (4, 5):
            if self._rtk_loss_since is None:
                self._rtk_loss_since = now
            lost_for = now - self._rtk_loss_since
            if lost_for >= c.rtk_loss_grace_s:
                self.state.phase = "abort"
                self._stop(f"not RTK for {lost_for:.2f}s (fix={fix['fix_type']})")
                return self._telemetry(fix, None, 0.0)
            return self._pause_for_signal(f"RTK dropout {lost_for:.2f}s (fix={fix['fix_type']})", fix)
        self._rtk_loss_since = None

        pos = (fix["lat"], fix["lon"])

        # --- goal check ---
        dist_to_goal = geo.haversine_m(pos, self.path[-1])
        near_last = geo.nearest_index(self.path, pos) >= len(self.path) - 1
        if near_last and dist_to_goal <= c.goal_radius_m:
            self.state.phase = "done"
            self._stop("arrived at goal", brake=c.arrival_brake)
            return self._telemetry(fix, None, 0.0, brake=c.arrival_brake)

        # --- heading-free cross-track tracking ---
        self.state.phase = "tracking"
        snap = geo.nearest_point_on_path(self.path, pos)

        # cross-track abort
        xtrack = snap.distance_m
        if xtrack > c.max_crosstrack_m:
            self.state.phase = "abort"
            self._stop(f"off path ({xtrack:.1f} m > {c.max_crosstrack_m} m)")
            return self._telemetry(fix, None, 0.0)

        # Signed cross-track error (+ = cart is LEFT of the path direction).
        cross = snap.signed_distance_m

        # Direction the purple line is heading at the nearest segment (compass).
        seg_i = snap.segment_index
        a = self.path[seg_i]
        b = self.path[min(seg_i + 1, len(self.path) - 1)]
        path_bearing = geo.bearing_deg(a, b)

        # Cart heading from the WHEELS, not GPS (per the operator's spec). We take
        # the measured steering angle — the same value shown as the panel's
        # "actual" — and convert it to an equivalent heading offset from the path.
        # This ONE quantity is the single source of truth: it feeds the steering
        # align term below AND the map needle, so the two can never disagree.
        # The needle shows this angle 1:1 (almost exactly the wheel angle); the
        # steering term uses a scaled-down copy so its feel/tuning is unchanged.
        # Works at any speed, unlike the old GPS estimate that was meaningless
        # below walking pace.
        # wheel_deg is kept only for the MAP NEEDLE (heading_abs below): the
        # needle leans by the measured wheel angle 1:1. It is NOT used for the
        # steering law any more — see the heading_err below.
        wheel_deg = self._smooth_heading(self._last_actual_steer_deg or 0.0)

        # Heading error, heading-FREE, from the cross-track RATE. Since
        # d(cross)/dt = v * sin(heading_err), we recover heading_err ≈
        # asin(cross_rate / v) from position + speed alone — no compass, works
        # the same whether the wheel is slewing or settled. (+ heading_err =
        # cart pointing LEFT of the path.) This is the damping term the law was
        # always meant to have (module docstring: "PD on (cross, cross_rate)");
        # the old wheel-angle "align" term carried no real trajectory damping,
        # so the loop limit-cycled — full lock across the line and back.
        now = time.time()
        cross_rate = self._cross_rate(cross, now)
        speed_ms = max(c.live_speed_mph * 0.44704, 0.3)
        raw_herr = math.degrees(math.asin(max(-1.0, min(1.0, cross_rate / speed_ms))))
        heading_err = self._smooth_heading_err(raw_herr)
        if c.live_speed_mph < c.heading_min_speed_mph:
            heading_err = 0.0   # rate estimate is meaningless when barely moving

        # Stanley-style PD (POSITIVE column command = RIGHT turn):
        #   pull  — P on cross-track: aim back toward the line, bounded by atan2.
        #   align — D (damping): cancel heading error so we arrive PARALLEL to
        #           the line instead of slicing across it and weaving back.
        # Right of the line (cross<0) -> pull<0 -> turn left. As we swing left to
        # recover, heading_err>0 (pointing left) -> align>0 -> turn right, easing
        # off the lock BEFORE we reach the line so we settle instead of overshoot.
        pull_deg = math.degrees(math.atan2(c.xtrack_gain * cross, max(c.lookahead_m, 0.5)))
        align_deg = c.heading_gain * heading_err
        correction_deg = pull_deg + align_deg
        raw_steer_deg = c.steer_sign * c.steer_gain * correction_deg
        steer_deg = max(-c.max_steer_deg, min(raw_steer_deg, c.max_steer_deg))

        # Map needle = the path bearing leaned by the wheel angle, 1:1 — so the
        # needle points almost exactly where the steering wheel is turned. Same
        # smoothed wheel reading the steering term uses above, just shown
        # unscaled. (GpsMarker eases the motion render-side; that doesn't change
        # this value.)
        heading_abs = (path_bearing + wheel_deg) % 360.0

        # --- throttle: closed-loop speed control (PI on GPS mph + brake) ---
        # Open-loop gas is miscalibrated (0.24 gave only ~2 mph) and blind to
        # grade, so we feed the GPS speed back: a feedforward guess gets us in
        # the ballpark, the integral adds gas to hold speed up a climb, and the
        # brake bleeds overspeed on a descent. The setpoint is max_speed_mph;
        # gas_cap is just the safety ceiling the controller may push to.
        now = time.time()
        dt_spd = (now - self._prev_speed_ts) if self._prev_speed_ts is not None else 0.0
        dt_spd = max(0.0, min(dt_spd, 1.0))   # ignore long gaps (startup/stalls)
        self._prev_speed_ts = now

        # Final-approach taper: ramp the setpoint linearly down to the creep
        # speed across arrival_slowdown_m so we decelerate smoothly into the
        # goal. The overspeed-brake term below bleeds the excess as the setpoint
        # drops; the goal cutoff above then holds the firm parking brake.
        target_mph = c.max_speed_mph
        if dist_to_goal < c.arrival_slowdown_m:
            frac = dist_to_goal / max(c.arrival_slowdown_m, 0.1)
            target_mph = max(c.arrival_creep_mph, c.max_speed_mph * frac)

        speed_error = target_mph - c.live_speed_mph          # + => want to speed up
        ff_gas = config.gas_for_mph(target_mph)               # open-loop ballpark
        i_cand = self._speed_integral + speed_error * dt_spd
        i_cand = max(-c.speed_i_max, min(i_cand, c.speed_i_max))

        gas_cmd = ff_gas + c.speed_kp * speed_error + c.speed_ki * i_cand
        gas_cmd /= (1.0 + c.turn_slowdown * abs(steer_deg) / max(c.max_steer_deg, 1.0))
        applied_gas = max(0.0, min(gas_cmd, c.gas_cap))

        # Conditional anti-windup: only commit the integral step when it isn't
        # pushing further into a saturated pedal (cap reached / gas already 0).
        sat_high = gas_cmd >= c.gas_cap and speed_error > 0
        sat_low = gas_cmd <= 0.0 and speed_error < 0
        if not (sat_high or sat_low):
            self._speed_integral = i_cand

        # Brake only once gas is cut and we're still over target past the
        # deadband (downhill); proportional to the overspeed.
        applied_brake = 0.0
        over = -speed_error - c.brake_deadband_mph
        if over > 0.0:
            applied_gas = 0.0
            applied_brake = min(c.brake_kp * over, config.BRAKE_POT_MAX)

        self._apply(gas=applied_gas, brake=applied_brake, steer_deg=steer_deg)
        return self._telemetry(fix, correction_deg, applied_gas, steer_deg,
                               snap.segment_index, xtrack, dist_to_goal, cross,
                               heading_abs, round(heading_err, 1), applied_brake)

    def _telemetry(self, fix, alpha, gas, steer_deg=0.0, look_i=-1,
                   xtrack=0.0, dist_to_goal=0.0, cross=None,
                   heading_deg=None, heading_err_deg=None, brake=0.0) -> dict:
        steering_actual = self._last_actual_steer_deg
        steering_target = None
        if self.cart.steering:
            steering_target = getattr(self.cart.steering, "target_deg", None)
            now = time.time()
            if now - self._last_steer_read_ts >= 0.25:
                self._last_steer_read_ts = now
                try:
                    steering_actual = self.cart.steering.angle_deg()
                    self._last_actual_steer_deg = steering_actual
                except Exception:
                    pass
        return {
            "phase": self.state.phase,
            "reason": self.state.reason,
            "fix": fix["fix_type"] if fix else None,
            "lat": fix["lat"] if fix else None,
            "lon": fix["lon"] if fix else None,
            "ts": fix["ts"] if fix else None,
            "alpha": round(alpha, 1) if alpha is not None else None,
            "steer_cmd": round(steer_deg, 1),
            "gas": round(gas, 3),
            "brake": round(brake, 3),
            "max_speed_mph": round(self.cfg.max_speed_mph, 1),
            "lookahead_i": look_i,
            "xtrack_m": round(xtrack, 2),
            "xtrack_signed_m": round(cross, 2) if cross is not None else None,
            "heading_deg": round(heading_deg, 1) if heading_deg is not None else None,
            "heading_err_deg": heading_err_deg,
            "dist_to_goal_m": round(dist_to_goal, 1),
            "steering_actual_deg": round(steering_actual, 1) if steering_actual is not None else None,
            "steering_target_deg": round(steering_target, 1) if steering_target is not None else None,
            "live_speed_mph": round(self.cfg.live_speed_mph, 1),
            "lookahead_m": round(self.cfg.lookahead_m, 2),
            "steer_gain": round(self.cfg.steer_gain, 2),
            "xtrack_gain": round(self.cfg.xtrack_gain, 2),
            "max_steer_deg": round(self.cfg.max_steer_deg, 1),
            "turn_slowdown": round(self.cfg.turn_slowdown, 2),
            "armed": self.armed,
        }

    # -- run loop ----------------------------------------------------------
    def run(self, on_step=None) -> str:
        """Drive the loop until done/abort. Returns the terminating phase.

        ``on_step(telemetry)`` is called every cycle (use it to print/log).
        Always releases the throttle on exit.
        """
        period = 1.0 / self.cfg.rate_hz
        try:
            while True:
                tele = self.step()
                if on_step:
                    on_step(tele)
                if self.state.phase in ("done", "abort"):
                    return self.state.phase
                time.sleep(period)
        finally:
            if self.armed and self.cart.pedals:
                self.cart.pedals.set_gas(0.0)
                self.cart.pedals.stop()
