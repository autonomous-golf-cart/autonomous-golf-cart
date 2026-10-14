"""Steering prediction from the planned BEV path, plus ground-truth helpers
for the recorded control log."""
import json
import math

STEERING_RATIO = 6.8  # column degrees per wheel degree


class SteeringEstimator:
    """Real-time causal steering estimator.

    Uses only BEV lane-center path heading (vision-only, no IMU curvature).
    EMA smoothing accumulates temporal evidence across frames.

    Tuned via Ridge regression (5-fold CV MAE ≈ 28.0°):
      steering ≈ W_BEV × EMA(bev_heading) + intercept

    Call update_bev() when a new BEV frame is available (~2 Hz).
    Read .steering_deg for the current prediction.
    """

    BEV_EMA = 0.30          # EMA alpha for BEV heading — balances responsiveness vs noise
    W_BEV = 142.0           # regression weight: radians → steering degrees
    INTERCEPT = 9.0         # regression intercept

    def __init__(self):
        self._ema_heading = 0.0

    @property
    def steering_deg(self):
        raw = self.W_BEV * self._ema_heading + self.INTERCEPT
        return max(-270.0, min(270.0, raw))

    def update_bev(self, lane_local):
        """Feed BEV lane-center path in ego-local coords (fwd, left).

        Extracts path heading in the 0.5–3 m range ahead of the ego
        and applies causal EMA smoothing.
        """
        if lane_local is None or len(lane_local) < 4:
            return
        fwd = lane_local[:, 0]
        left = lane_local[:, 1]
        mask = (fwd > 0.5) & (fwd < 3.0)
        if mask.sum() >= 2:
            raw_heading = math.atan2(
                left[mask][-1] - left[mask][0],
                fwd[mask][-1] - fwd[mask][0],
            )
            self._ema_heading = (self.BEV_EMA * raw_heading +
                                 (1 - self.BEV_EMA) * self._ema_heading)


def load_control_data(control_path):
    controls = []
    with open(control_path) as f:
        for line in f:
            c = json.loads(line.strip())
            controls.append(c)
    return controls


def get_gt_steering_at_time(controls, t):
    for i in range(len(controls) - 1):
        if controls[i]["rel_t"] <= t <= controls[i + 1]["rel_t"]:
            alpha = (t - controls[i]["rel_t"]) / (controls[i + 1]["rel_t"] - controls[i]["rel_t"] + 1e-9)
            return controls[i]["column_deg_actual"] * (1 - alpha) + controls[i + 1]["column_deg_actual"] * alpha
    return controls[-1]["column_deg_actual"]
