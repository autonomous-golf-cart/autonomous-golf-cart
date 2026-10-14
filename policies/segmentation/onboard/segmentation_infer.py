#!/usr/bin/env python3
"""
segmentation_infer.py — drive-by-segmentation live planner sidecar.

Uses the segdrive package in the parent directory for SegFormer semantic segmentation, BEV projection,
lane-aware trajectory planning, and steering estimation. Publishes the
same /tmp contract as the other model sidecars:

  /tmp/cart_frames/{front_narrow,left,front_wide,right}.jpg  raw cameras
  /tmp/cart_frames/seg.jpg                                  segmentation overlay
  /tmp/cart_frames/bev.jpg                                  BEV trajectory
  /tmp/autoware_state.json                                  steer + pedal targets

start.sh launches this with --model segmentation. ps5_drive.py --autosteer
then reads steer_deg plus target_gas/target_brake and applies the usual
operator trigger overrides.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seg_fast  # noqa: E402


SEG_REPO_DEFAULT = Path(__file__).resolve().parent.parent  # policies/segmentation/
FRAMES_DIR_DEFAULT = Path("/tmp/cart_frames")
STATE_FILE_DEFAULT = Path("/tmp/autoware_state.json")
EGO_STATE_FILE_DEFAULT = Path(os.environ.get("EGO_STATE_FILE", "/tmp/ego_state.json"))

# Match the usual live-model camera labels on this cart. The second and
# third discovered cameras are front_wide / left respectively; if these are
# reversed, segmentation will infer from the side-facing left camera.
SLUGS = ("front_narrow", "front_wide", "left", "right")
ACTIVE_SLUG = "front_wide"
CAMERA_ORIENTATION_FIX = {"front_narrow": -1}

CAM_W, CAM_H = 640, 480
JPEG_QUALITY = 72
PUBLISH_HZ_DEFAULT = 15.0
INFER_HZ_DEFAULT = 0.0
MODEL_NAME = "segmentation"
STEERING_SIGN = -1.0
STEERING_COLUMN_RATIO = 15.0
STEERING_EMA = 0.35
LOOKAHEAD_MIN_M = 2.5
LOOKAHEAD_MAX_M = 10.0
LOOKAHEAD_TIME_S = 0.9
# Scalar gain on the pure-pursuit road-wheel angle. PP is geometrically
# correct but tends to feel under-geared on a short-wheelbase low-speed
# kart; bump above 1.0 for more aggressive turn-in.
STEER_GAIN = 1.6

# Pure-pursuit on an arc-length-parameterized BEV centerline.
# WHEELBASE_M : kart rear-to-front axle. ~0.8 m placeholder; tune live.
# CENTERLINE_SMOOTH_WIN : window (in lane_local samples) for rolling-mean
#   smoothing of the centerline before arc-length resampling. Tames per-
#   frame planner jitter without lagging genuine path bends.
# LOOKAHEAD_POINT_EMA : EMA on the (xL, yL) lookahead point itself. Smooths
#   point jitter without adding lag to the steering response — preferred
#   over cranking STEERING_EMA, which would delay sharp turns.
WHEELBASE_M = 0.8
CENTERLINE_SMOOTH_WIN = 5
LOOKAHEAD_POINT_EMA = 0.3

# Limit pure-pursuit input to a near window — the far end of the planner's
# polyline is too noisy to inform the lookahead point reliably.
STEER_FIT_MIN_M = 1.0
STEER_FIT_MAX_M = 10.0

# Constant-speed target. ps5_drive consumes absolute pedal pot targets, so
# keep this conservative and publish the intended speed separately for UI/logs.
TARGET_MPH = 8.0
GAS_CONSTANT = 0.24
GAS_PER_MPH = GAS_CONSTANT / TARGET_MPH
BRAKE_CONSTANT = 0.0

# Closed-loop speed control: PI on (target_mph − ego_mph) when ARKit is
# fresh. Output is added to the open-loop feed-forward gas. Conservative
# gains — we'd rather creep up than overshoot into the cart's natural
# spring-back. Reset integrator when ARKit drops to avoid wind-up.
SPEED_KP = 0.03   # gas per mph error
SPEED_KI = 0.02   # gas per (mph·s) error
SPEED_I_CLAMP = 0.30  # safety cap on |integral term| (gas units)
# Trim clamp scales with the feed-forward so behavior is consistent across
# target speeds — at 2 mph (ff≈0.06) we allow ±0.10 trim, at 8 mph (ff≈0.24)
# we allow ±0.36. Prevents the wild 4× swing the fixed ±0.25 caused at low mph.
GAS_TRIM_FLOOR = 0.20
GAS_TRIM_SCALE = 1.5
# Rolling-resistance floor: below this gas the kart barely moves. The pot↔mph
# mapping (GAS_PER_MPH) is calibrated against the linear 5–8 mph regime; at
# low mph the kart needs more gas than the linear extrapolation suggests just
# to overcome rolling friction. Force feed-forward to at least this when the
# user actually wants to move.
ROLLING_GAS_FLOOR = 0.18

# Brake-on-overshoot: when ego > target AND gas has already hit 0 (cannot
# trim further), engage brake proportional to overshoot. Brake and gas are
# mutually exclusive — coast-then-brake, never both — to avoid pad wear
# under normal cruise. Only activates after BRAKE_DEADBAND_MPH of overshoot
# so small ARKit jitter doesn't pulse the brake.
BRAKE_KP = 0.06           # brake per mph overshoot
BRAKE_MAX = 0.25          # absolute brake-pot ceiling under autosteer
BRAKE_DEADBAND_MPH = 0.3  # mph overshoot before brake engages

# Launch ramp: start at LAUNCH_GAS_MIN and ease up to the commanded gas
# over LAUNCH_RAMP_S seconds so the kart doesn't jerk off the line. The
# ramp is applied to the *final* commanded gas (ff + PI trim), so the PI
# loop still corrects within the ramp ceiling instead of fighting it.
LAUNCH_RAMP_S = 3.75
LAUNCH_GAS_MIN = 0.02
# Below this gas the kart can't break static friction. Ramp always rises to
# at least this value (regardless of target_mph), and the stuck-detector
# punches gas to it if we haven't started moving after the ramp ends. Once
# the kart is rolling, the PI naturally trims back down to feed-forward.
STICTION_GAS_BREAK = 0.22
STICTION_EGO_MPH = 0.3
STICTION_STUCK_S = 1.0


def discover_v4l2_indices(count: int = 4, max_scan: int = 16) -> list[int]:
    found: list[int] = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
            if len(found) >= count:
                break
    return found


def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class RealSenseReader(threading.Thread):
    """RealSense color + aligned depth source.

    Mirrors CameraReader's `.latest()` / `.stop()` API so it slots into the
    existing slug_to_reader map. `latest_color_depth()` additionally returns
    the matched depth frame and pinhole intrinsics needed for the BEV cloud.
    """

    def __init__(self, slug: str, width: int = 640, height: int = 480,
                 fps: int = 30, enable_depth: bool = True):
        super().__init__(daemon=True, name=f"cam-{slug}")
        import pyrealsense2 as rs
        self.rs = rs
        self.slug = slug
        self.enable_depth = enable_depth
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        if enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
        else:
            self.depth_scale = 0.0
            self.align = None
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_prof.get_intrinsics()
        self.intrinsics = (float(intr.fx), float(intr.fy),
                           float(intr.ppx), float(intr.ppy))
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(2000)
            except Exception:
                time.sleep(0.05)
                continue
            if self.enable_depth:
                aligned = self.align.process(frames)
                c = aligned.get_color_frame()
                d = aligned.get_depth_frame()
                if not c or not d:
                    continue
                color = np.asanyarray(c.get_data())
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
                if self.flip_code is not None:
                    color = cv2.flip(color, self.flip_code)
                    depth = cv2.flip(depth, self.flip_code)
            else:
                c = frames.get_color_frame()
                if not c:
                    continue
                color = np.asanyarray(c.get_data())
                depth = None
                if self.flip_code is not None:
                    color = cv2.flip(color, self.flip_code)
            with self.lock:
                self.frame = color
                self.depth = depth
                self.frame_count += 1
                self.last_ok_s = time.monotonic()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_color_depth(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        with self.lock:
            if self.frame is None:
                return None
            if not self.enable_depth:
                return self.frame.copy(), None
            if self.depth is None:
                return None
            return self.frame.copy(), self.depth.copy()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.pipeline.stop()
        except Exception:
            pass


class CameraReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, slug: str):
        super().__init__(daemon=True, name=f"cam-{slug}")
        self.cap = cap
        self.slug = slug
        self.flip_code = CAMERA_ORIENTATION_FIX.get(slug)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.flip_code is not None:
                    frame = cv2.flip(frame, self.flip_code)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self.cap.release()


class VideoReader(threading.Thread):
    def __init__(self, video_path: str, loop: bool = True):
        super().__init__(daemon=True, name="video-reader")
        self.video_path = video_path
        self.loop = loop
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_ok_s = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[video] failed to open {self.video_path}", flush=True)
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            period = 1.0 / fps
            next_t = time.monotonic()
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_ok_s = time.monotonic()
                next_t += period
                wait = next_t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            cap.release()
            if not self.loop:
                return

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self._stop.set()


def write_jpeg_atomic(path: Path, frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def sync_cuda_if_needed(device: str) -> None:
    if device != "cuda":
        return
    import torch
    torch.cuda.synchronize()


def timed_segment_frame(frame_rgb: np.ndarray, proc, model, device: str) -> tuple[np.ndarray, dict[str, float]]:
    import torch

    timings: dict[str, float] = {}
    t = time.perf_counter()
    pil = Image.fromarray(frame_rgb)
    timings["seg_pil_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    inputs = proc(images=pil, return_tensors="pt")
    timings["seg_preprocess_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    inputs = inputs.to(device)
    dtype = next(model.parameters()).dtype
    if inputs["pixel_values"].dtype != dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    sync_cuda_if_needed(device)
    timings["seg_transfer_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    sync_cuda_if_needed(device)
    timings["seg_forward_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    seg = proc.post_process_semantic_segmentation(
        out, target_sizes=[frame_rgb.shape[:2]]
    )[0].cpu().numpy().astype(np.uint8)
    sync_cuda_if_needed(device)
    timings["seg_postprocess_ms"] = (time.perf_counter() - t) * 1000.0
    return seg, timings


def read_ego_speed_mph(path: Path, fresh_s: float = 1.0) -> tuple[float, bool]:
    try:
        with path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0, False
    if time.time() - float(data.get("ts", 0.0)) > fresh_s:
        return 0.0, False
    if not data.get("connected", False):
        return 0.0, False
    sample = data.get("sample") or {}
    try:
        speed_mps = abs(float(sample.get("speed_mps", 0.0)))
    except (TypeError, ValueError):
        return 0.0, False
    return speed_mps * 2.23694, True


def adaptive_lookahead_m(speed_mph: float, speed_ok: bool) -> float:
    if not speed_ok:
        speed_mph = 2.0
    speed_mps = speed_mph * 0.44704
    return float(np.clip(speed_mps * LOOKAHEAD_TIME_S, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M))


def constant_gas_for_mph(target_mph: float) -> float:
    # Linear pot↔mph map, but raised to ROLLING_GAS_FLOOR whenever the user
    # asks for any forward motion. Without the floor, low-mph targets sit
    # under the kart's actual rolling-resistance gas and the loop tops out
    # short of the setpoint.
    if target_mph <= 0.0:
        return 0.0
    return float(np.clip(max(target_mph * GAS_PER_MPH, ROLLING_GAS_FLOOR), 0.0, 1.0))


class SpeedController:
    """PI on (target_mph - ego_mph), output added to the open-loop gas."""

    def __init__(self) -> None:
        self.integral = 0.0
        self.last_t = None

    def reset(self) -> None:
        self.integral = 0.0
        self.last_t = None

    def step(self, target_mph: float, ego_mph: float, ego_ok: bool,
             feed_forward_gas: float, gas_ceiling: float = 1.0) -> tuple[float, float, float]:
        """Return (gas, trim, brake). Gas/brake are mutually exclusive.

        `gas_ceiling` lets the launch ramp cap the output without winding
        the integrator: when the ramp is clipping us low, we suppress
        integration in the direction that would wind further into the
        clip — classic conditional-integration anti-windup.
        """
        now = time.monotonic()
        if not ego_ok:
            self.reset()
            return float(np.clip(feed_forward_gas, 0.0, gas_ceiling)), 0.0, 0.0
        dt = 0.1 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        err = target_mph - ego_mph

        # Trim clamp scales with feed-forward → consistent authority at any
        # target speed. At 2 mph: ±0.10 ; at 8 mph: ±0.36.
        trim_clamp = max(GAS_TRIM_FLOOR, GAS_TRIM_SCALE * feed_forward_gas)

        # Provisional integral, with conditional anti-windup:
        # don't integrate further into the direction we're already saturated.
        trial_integral = self.integral + err * dt
        trim_unclipped = SPEED_KP * err + SPEED_KI * trial_integral
        gas_unclipped = feed_forward_gas + trim_unclipped
        saturated_high = gas_unclipped > gas_ceiling
        saturated_low = gas_unclipped < 0.0
        winding_into_clip = (
            (saturated_high and err > 0) or (saturated_low and err < 0)
        )
        if not winding_into_clip:
            self.integral = float(np.clip(
                trial_integral,
                -SPEED_I_CLAMP / max(SPEED_KI, 1e-6),
                 SPEED_I_CLAMP / max(SPEED_KI, 1e-6),
            ))

        trim = SPEED_KP * err + SPEED_KI * self.integral
        trim = float(np.clip(trim, -trim_clamp, trim_clamp))
        gas = float(np.clip(feed_forward_gas + trim, 0.0, gas_ceiling))

        # Engine braking via gas-off comes "free" — only blend in pad brake
        # when we're still over target *with gas already at zero*. This keeps
        # cruise running on gas alone and only touches the pedal pad when
        # gravity (downhill) wants to push us past target.
        brake = 0.0
        overshoot = ego_mph - target_mph - BRAKE_DEADBAND_MPH
        if gas <= 1e-3 and overshoot > 0.0:
            brake = float(np.clip(BRAKE_KP * overshoot, 0.0, BRAKE_MAX))
        return gas, trim, brake


_lookahead_point_filt: tuple[float, float] | None = None


def reset_lookahead_heading_filter() -> None:
    global _lookahead_point_filt
    _lookahead_point_filt = None


def _smooth_centerline(pts: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(pts) < win:
        return pts
    k = np.ones(win, dtype=np.float64) / win
    xs = np.convolve(pts[:, 0], k, mode="same")
    ys = np.convolve(pts[:, 1], k, mode="same")
    return np.stack([xs, ys], axis=1)


def _resample_lookahead_point(pts: np.ndarray, lookahead_m: float) -> tuple[float, float] | None:
    """Interpolate the lane point at arc length ≈ lookahead_m.

    Works for arbitrarily sharp turns because arc length grows monotonically
    even when the path curls past 90° (where y(x) breaks down)."""
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return None
    s_target = min(float(lookahead_m), float(s[-1]))
    xL = float(np.interp(s_target, s, pts[:, 0]))
    yL = float(np.interp(s_target, s, pts[:, 1]))
    return xL, yL


def lookahead_heading_steering_deg(lane_local: np.ndarray, lookahead_m: float) -> float | None:
    """Pure-pursuit steering from a BEV lane centerline.

    1. Smooth lane_local with a rolling mean (kills planner jitter).
    2. Resample by arc length to get a stable lookahead point P=(xL, yL) —
       this works at any turn sharpness, unlike a y(x) polynomial fit.
    3. EMA the lookahead point (not the angle) so jitter is removed without
       lagging real steering response.
    4. Apply the bicycle pure-pursuit law:
           alpha = atan2(yL, xL)                          # bearing, ±π
           delta = atan2(2 L sin(alpha), Ld)              # road-wheel rad
       alpha ranges across the full ±π, so delta * STEERING_COLUMN_RATIO
       can ride all the way to ±270°.
    """
    global _lookahead_point_filt

    if lane_local is None or len(lane_local) < 4:
        return None

    pts = np.asarray(lane_local, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 0] > STEER_FIT_MIN_M) & (pts[:, 0] < STEER_FIT_MAX_M)]
    if len(pts) < 4:
        return None

    pts = _smooth_centerline(pts, CENTERLINE_SMOOTH_WIN)
    p = _resample_lookahead_point(pts, lookahead_m)
    if p is None:
        return None
    xL, yL = p

    if _lookahead_point_filt is None:
        _lookahead_point_filt = (xL, yL)
    else:
        a = LOOKAHEAD_POINT_EMA
        _lookahead_point_filt = (
            a * xL + (1.0 - a) * _lookahead_point_filt[0],
            a * yL + (1.0 - a) * _lookahead_point_filt[1],
        )
    xLf, yLf = _lookahead_point_filt

    Ld = math.hypot(xLf, yLf)
    if Ld < 1e-3:
        return None
    alpha = math.atan2(yLf, xLf)
    delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), Ld)
    column_deg = math.degrees(delta) * STEERING_COLUMN_RATIO * STEER_GAIN
    return float(np.clip(column_deg, -270.0, 270.0))


def make_sources(args) -> list[CameraReader | VideoReader | RealSenseReader]:
    if args.video:
        reader = VideoReader(args.video, loop=not args.no_loop)
        reader.slug = ACTIVE_SLUG
        return [reader]

    if args.source == "realsense":
        rs_reader = RealSenseReader(
            slug=ACTIVE_SLUG,
            width=args.rs_width, height=args.rs_height, fps=args.rs_fps,
            enable_depth=not args.no_depth,
        )
        return [rs_reader]

    indices = discover_v4l2_indices(count=len(SLUGS), max_scan=args.max_scan)
    if len(indices) < len(SLUGS):
        raise RuntimeError(f"expected {len(SLUGS)} cameras, found {indices}")

    readers: list[CameraReader] = []
    for slug, idx in zip(SLUGS, indices):
        cap = open_camera(idx)
        if cap is None:
            raise RuntimeError(f"failed to open camera {idx} for {slug}")
        readers.append(CameraReader(cap, slug))
    return readers


def import_segmentation_stack(repo_dir: Path):
    if not (repo_dir / "segdrive").is_dir():
        raise RuntimeError(f"segdrive package not found in {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    from segdrive import draw as rt
    from segdrive import segformer as seg_live
    from segdrive.bev import create_bev, create_overlay
    from segdrive.cityscapes import CITYSCAPES_COLORS
    from segdrive.planning import lane_aware_centerline_path

    return seg_live, rt, lane_aware_centerline_path, create_bev, create_overlay, CITYSCAPES_COLORS


def fill_bev_holes(bev_rgb: np.ndarray, mode: str = "fast-inpaint",
                   radius: int = 3, iterations: int = 3,
                   scale: int = 4) -> np.ndarray:
    """Fill empty BEV pixels.

    The old cv2.INPAINT_TELEA path is visually smooth but costs ~150 ms on
    Thor for a 500x500 BEV and can create blended colors that no longer match
    Cityscapes classes exactly. The default dilation path is intentionally
    cheaper: it copies nearby class-colored pixels into small gaps and keeps
    exact palette colors for the planner's road mask.
    """
    if bev_rgb.size == 0:
        return bev_rgb
    empty = np.all(bev_rgb == 0, axis=-1).astype(np.uint8) * 255
    if empty.max() == 0:
        return bev_rgb
    if mode == "none":
        return bev_rgb
    if mode == "inpaint":
        return cv2.inpaint(bev_rgb, empty, radius, cv2.INPAINT_TELEA)
    if mode == "fast-inpaint":
        scale = max(1, int(scale))
        if scale == 1:
            return cv2.inpaint(bev_rgb, empty, radius, cv2.INPAINT_TELEA)
        h, w = bev_rgb.shape[:2]
        small_w = max(1, w // scale)
        small_h = max(1, h // scale)
        small = cv2.resize(bev_rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        small_empty = np.all(small == 0, axis=-1).astype(np.uint8) * 255
        if small_empty.max() != 0:
            small = cv2.inpaint(small, small_empty, radius, cv2.INPAINT_TELEA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "nearest":
        filled = np.any(bev_rgb != 0, axis=-1)
        if not np.any(filled):
            return bev_rgb
        dt_src = (~filled).astype(np.uint8) * 255
        _, labels = cv2.distanceTransformWithLabels(
            dt_src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        colors = bev_rgb[filled]
        label_colors = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        if len(colors) >= labels.max():
            label_colors[1:] = colors[: labels.max()]
        else:
            label_colors[labels[filled]] = colors
        return label_colors[labels]

    out = bev_rgb.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    remaining = empty.astype(bool)
    for _ in range(max(0, int(iterations))):
        dilated = cv2.dilate(out, kernel, iterations=1)
        filled_now = remaining & np.any(dilated != 0, axis=-1)
        if not np.any(filled_now):
            break
        out[filled_now] = dilated[filled_now]
        remaining[filled_now] = False
        if not np.any(remaining):
            break
    return out


def cloud_bev_rgb(depth_m: np.ndarray, seg_map: np.ndarray,
                  intrinsics: tuple[float, float, float, float],
                  bev_size: int, range_fwd_m: float, range_side_m: float,
                  palette: np.ndarray, min_depth_m: float = 0.3,
                  max_depth_m: float = 80.0, stride: int = 2,
                  splat: int = 1, fill_holes: bool = True,
                  fill_mode: str = "fast-inpaint", inpaint_radius: int = 3,
                  fill_iterations: int = 3, fill_scale: int = 4) -> np.ndarray:
    """Render a top-down BEV by unprojecting depth pixels via pinhole intrinsics
    and splatting each onto a `bev_size x bev_size` canvas, colored by seg class.

    Output is RGB, ego at bottom-center, forward up, half-width = range_side_m.
    Compatible with the homography BEV's coordinate convention so plan_path /
    SteeringEstimator / lookahead all keep working.
    """
    fx, fy, cx, cy = intrinsics
    stride = max(1, int(stride))
    h, w = depth_m.shape
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    if seg_map.shape != (h, w):
        seg_map = cv2.resize(seg_map.astype(np.int32), (w, h),
                             interpolation=cv2.INTER_NEAREST)

    if stride > 1:
        depth_sample = depth_m[::stride, ::stride]
        mask = (
            np.isfinite(depth_sample)
            & (depth_sample >= min_depth_m)
            & (depth_sample <= max_depth_m)
        )
        sv, su = np.nonzero(mask)
        if sv.size == 0:
            return bev
        v = sv * stride
        u = su * stride
        z = depth_sample[sv, su]
    else:
        mask = (
            np.isfinite(depth_m)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
        )
        v, u = np.nonzero(mask)
        if v.size == 0:
            return bev
        z = depth_m[v, u]

    x = (u.astype(np.float32) - cx) * z / fx
    cls = np.clip(seg_map[v, u], 0, len(palette) - 1)
    colors = palette[cls]

    keep = (z > 0) & (z <= range_fwd_m) & (np.abs(x) <= range_side_m)
    if not np.any(keep):
        return bev
    x = x[keep]; z = z[keep]; colors = colors[keep]
    px = ((x / range_side_m * 0.5 + 0.5) * (bev_size - 1)).astype(np.int32)
    py = ((1.0 - z / range_fwd_m) * (bev_size - 1)).astype(np.int32)

    order = np.argsort(z)[::-1]
    px = px[order]; py = py[order]; colors = colors[order]

    if splat <= 0:
        bev[py, px] = colors
    else:
        for dy in range(-splat, splat + 1):
            for dx in range(-splat, splat + 1):
                yy = np.clip(py + dy, 0, bev_size - 1)
                xx = np.clip(px + dx, 0, bev_size - 1)
                bev[yy, xx] = colors

    if fill_holes:
        bev = fill_bev_holes(
            bev,
            mode=fill_mode,
            radius=inpaint_radius,
            iterations=fill_iterations,
            scale=fill_scale,
        )
    return bev


def draw_bev_viz(bev_rgb: np.ndarray, lane_traj: np.ndarray | None,
                 lane_local: np.ndarray | None, rt) -> np.ndarray:
    out = bev_rgb.copy()
    # Foot-labelled distance grid. RANGE_FWD covers 0..forward_ft (ego at
    # bottom); RANGE_SIDE covers ±side_ft (ego column at horizontal
    # center). Minor lines every 5 ft, major + label every 10 ft.
    FT_PER_M = 3.28084
    fwd_ft = rt.RANGE_FWD * FT_PER_M
    side_ft = rt.RANGE_SIDE * FT_PER_M
    h, w = out.shape[:2]
    ego_bx, ego_by = w // 2, h - 1
    minor_col = (90, 90, 90)
    major_col = (180, 180, 180)
    label_col = (255, 255, 255)
    label_shadow = (0, 0, 0)

    def put_label(img, text, org):
        cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_shadow, 2, cv2.LINE_AA)
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_col, 1, cv2.LINE_AA)

    # Forward ticks (horizontal lines), every 5 ft up to fwd_ft.
    for ft in range(5, int(fwd_ft) + 1, 5):
        x_m = ft / FT_PER_M
        by = int(round((1.0 - x_m / rt.RANGE_FWD) * h))
        if not (0 <= by < h):
            continue
        is_major = (ft % 10 == 0)
        cv2.line(out, (0, by), (w - 1, by),
                 major_col if is_major else minor_col, 1, cv2.LINE_AA)
        if is_major:
            put_label(out, f"{ft}ft", (ego_bx + 6, by - 3))

    # Lateral ticks (vertical lines), every 5 ft from -side_ft..+side_ft.
    side_max = int(side_ft)
    for ft in range(-side_max, side_max + 1, 5):
        if ft == 0:
            continue
        y_m = ft / FT_PER_M
        bx = int(round((y_m / rt.RANGE_SIDE * 0.5 + 0.5) * w))
        if not (0 <= bx < w):
            continue
        is_major = (ft % 10 == 0)
        cv2.line(out, (bx, 0), (bx, h - 1),
                 major_col if is_major else minor_col, 1, cv2.LINE_AA)
        if is_major:
            put_label(out, f"{ft:+d}ft", (bx + 3, ego_by - 6))

    # Axes through the ego.
    cv2.line(out, (ego_bx, 0), (ego_bx, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(out, (0, ego_by), (w - 1, ego_by), (220, 220, 220), 1, cv2.LINE_AA)
    put_label(out, "x=forward (ft)", (ego_bx + 6, 16))
    put_label(out, "y=lateral (ft)", (8, ego_by - 6))

    if lane_traj is not None:
        rt.draw_trajectory(out, lane_traj, (255, 255, 0), 3)
    la_point, _ = rt.lookahead_point(lane_local)
    if la_point is not None:
        la_bx = int((la_point[1] / rt.RANGE_SIDE * 0.5 + 0.5) * rt.BEV_SIZE)
        la_by = int((1 - la_point[0] / rt.RANGE_FWD) * rt.BEV_SIZE)
        if 0 <= la_bx < rt.BEV_SIZE and 0 <= la_by < rt.BEV_SIZE:
            cv2.line(out, (ego_bx, ego_by), (la_bx, la_by),
                     (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (la_bx, la_by), 6, (0, 255, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description="drive-by-segmentation WebUI sidecar")
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR_DEFAULT)
    p.add_argument("--state-file", type=Path, default=STATE_FILE_DEFAULT)
    p.add_argument("--seg-repo", type=Path, default=SEG_REPO_DEFAULT)
    p.add_argument("--model", default="b0", choices=("b0", "b2", "b5"))
    p.add_argument("--device", default=None)
    p.add_argument("--publish-hz", type=float, default=PUBLISH_HZ_DEFAULT)
    p.add_argument("--infer-hz", type=float, default=INFER_HZ_DEFAULT)
    p.add_argument("--video", default=None)
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--max-scan", type=int, default=16)
    p.add_argument("--source", default="uvc", choices=("uvc", "realsense"),
                   help="Active camera source.")
    p.add_argument("--bev-mode", default="homography", choices=("homography", "depth"),
                   help="BEV projection mode. 'homography' is the original "
                        "drive-by-segmentation calibrated ground-plane projection; "
                        "'depth' uses RealSense depth unprojection.")
    p.add_argument("--rs-width", type=int, default=640)
    p.add_argument("--rs-height", type=int, default=480)
    p.add_argument("--rs-fps", type=int, default=30)
    p.add_argument("--no-depth", action="store_true",
                   help="With --source realsense, skip the depth stream "
                        "(color only). Forces --bev-mode homography.")
    p.add_argument("--bev-splat", type=int, default=1,
                   help="Half-width (px) of the box used to splat each cloud point in the BEV.")
    p.add_argument("--bev-depth-stride", type=int, default=2,
                   help="Use every Nth RealSense depth pixel for BEV projection. "
                        "Higher values are faster but less dense.")
    p.add_argument("--no-bev-fill", action="store_true",
                   help="Disable inpaint-based hole filling on the cloud BEV.")
    p.add_argument("--bev-fill-mode", default="fast-inpaint",
                   choices=("fast-inpaint", "dilate", "nearest", "inpaint", "none"),
                   help="How to fill sparse cloud BEV holes. 'dilate' is fast; "
                        "'inpaint' is the old smooth but slow path.")
    p.add_argument("--bev-fill-iterations", type=int, default=3,
                   help="Number of 3x3 dilation passes for --bev-fill-mode dilate.")
    p.add_argument("--bev-fill-scale", type=int, default=4,
                   help="Downsample factor for --bev-fill-mode fast-inpaint.")
    p.add_argument("--bev-inpaint-radius", type=int, default=3,
                   help="cv2.inpaint radius used to fill empty BEV pixels.")
    p.add_argument("--profile-every", type=int, default=0,
                   help="Print one latency breakdown every N inference frames. "
                        "0 keeps profiling in state only.")
    p.add_argument("--ego-state-file", type=Path, default=EGO_STATE_FILE_DEFAULT)
    p.add_argument(
        "--target-mph",
        type=float,
        default=TARGET_MPH,
        help=f"Open-loop constant speed target in MPH (default {TARGET_MPH:g}).",
    )
    p.add_argument(
        "--no-fast", action="store_true",
        help="Use the original scipy-heavy create_bev + lane_aware_centerline_path "
             "instead of the cached / cv2-accelerated versions in seg_fast.",
    )
    args = p.parse_args()
    args.target_mph = max(0.0, float(args.target_mph))
    target_gas_ff = constant_gas_for_mph(args.target_mph)
    speed_ctrl = SpeedController()
    launch_start_t: float | None = None
    stuck_since: float | None = None

    seg_live, rt, plan_path, create_bev, create_overlay, colors = (
        import_segmentation_stack(args.seg_repo)
    )

    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with (args.seg_repo / "calibration" / "camera_calibration.json").open() as f:
        calib = json.load(f)

    rt.set_bev_range(calib)
    road_width_ft = float(calib.get("road_width_ft", 20.0))

    proc, model = seg_live.load_segformer(args.model, device)
    from segdrive.steering import SteeringEstimator
    steer_est = SteeringEstimator()

    readers = make_sources(args)
    for r in readers:
        r.start()

    slug_to_reader = {getattr(r, "slug", ACTIVE_SLUG): r for r in readers}
    active_reader = slug_to_reader.get(ACTIVE_SLUG)
    if active_reader is None:
        raise RuntimeError(f"active camera {ACTIVE_SLUG} not available")

    palette = np.array(colors, dtype=np.uint8)
    road_color = np.array(colors[0], dtype=np.uint8)
    grid_color = np.clip(road_color.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    grid2_color = np.clip(road_color.astype(np.int16) + 70, 0, 255).astype(np.uint8)

    publish_period = 1.0 / max(args.publish_hz, 1e-3)
    infer_period = 1.0 / args.infer_hz if args.infer_hz > 0 else 0.0
    next_publish_t = 0.0
    next_infer_t = 0.0
    infer_times: list[float] = []

    bev_remap: seg_fast.BevRemap | None = None
    use_fast = not args.no_fast and args.bev_mode == "homography"

    latest_overlay_bgr: np.ndarray | None = None
    latest_bev_bgr: np.ndarray | None = None
    latest_path: list[list[float]] = []
    latest_steer_raw = 0.0
    latest_steer_base = 0.0
    latest_steer_filtered = 0.0
    latest_lookahead_m = 0.0
    latest_ego_speed_mph = 0.0
    latest_ego_speed_ok = False
    latest_target_gas = target_gas_ff
    latest_gas_trim = 0.0
    latest_target_brake = 0.0
    inference_ok = False
    latest_latency_ms: dict[str, float] = {}
    infer_count = 0

    try:
        while True:
            now = time.monotonic()

            if now >= next_infer_t:
                stage_start = time.perf_counter()
                stage_last = stage_start
                stage_ms: dict[str, float] = {}

                def mark(name: str) -> None:
                    nonlocal stage_last
                    current = time.perf_counter()
                    stage_ms[name] = (current - stage_last) * 1000.0
                    stage_last = current

                if args.source == "realsense":
                    rs_pair = active_reader.latest_color_depth()
                    frame_bgr = rs_pair[0] if rs_pair is not None else None
                    depth_m = rs_pair[1] if rs_pair is not None else None
                else:
                    frame_bgr = active_reader.latest()
                    depth_m = None
                mark("source_latest_ms")
                if frame_bgr is not None:
                    t0 = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mark("bgr_to_rgb_ms")
                    seg_map, seg_stage_ms = timed_segment_frame(frame_rgb, proc, model, device)
                    stage_ms.update(seg_stage_ms)
                    stage_last = time.perf_counter()

                    if args.bev_mode == "depth":
                        if args.source != "realsense":
                            raise RuntimeError("--bev-mode depth requires --source realsense")
                        bev_rgb = cloud_bev_rgb(
                            depth_m, seg_map, active_reader.intrinsics,
                            bev_size=rt.BEV_SIZE,
                            range_fwd_m=rt.RANGE_FWD,
                            range_side_m=rt.RANGE_SIDE,
                            palette=palette,
                            stride=args.bev_depth_stride,
                            splat=args.bev_splat,
                            fill_holes=not args.no_bev_fill,
                            fill_mode="none" if args.no_bev_fill else args.bev_fill_mode,
                            inpaint_radius=args.bev_inpaint_radius,
                            fill_iterations=args.bev_fill_iterations,
                            fill_scale=args.bev_fill_scale,
                        )
                    else:
                        if use_fast:
                            if bev_remap is None:
                                img_h, img_w = frame_rgb.shape[:2]
                                bev_remap = seg_fast.build_bev_remap(
                                    calib, img_h, img_w, rt.BEV_SIZE
                                )
                                print(
                                    f"[seg] built BEV remap ({img_w}x{img_h} -> "
                                    f"{rt.BEV_SIZE}x{rt.BEV_SIZE}); cv2-accelerated "
                                    f"planner active.",
                                    flush=True,
                                )
                            bev_rgb = seg_fast.create_bev_cached(seg_map, palette, bev_remap)
                        else:
                            bev_rgb = create_bev(seg_map, calib, rt.BEV_SIZE)
                    mark("bev_ms")
                    road_mask = (
                        np.all(bev_rgb == road_color, axis=-1)
                        | np.all(bev_rgb == grid_color, axis=-1)
                        | np.all(bev_rgb == grid2_color, axis=-1)
                    )
                    mark("road_mask_ms")
                    if use_fast:
                        lane_traj, lane_local = seg_fast.lane_aware_centerline_path_fast(
                            road_mask,
                            bev_size=rt.BEV_SIZE,
                            range_fwd=rt.RANGE_FWD,
                            range_side=rt.RANGE_SIDE,
                            road_width_ft=road_width_ft,
                        )
                    else:
                        lane_traj, lane_local = plan_path(
                            road_mask,
                            bev_size=rt.BEV_SIZE,
                            range_fwd=rt.RANGE_FWD,
                            range_side=rt.RANGE_SIDE,
                            road_mask=road_mask,
                            road_width_ft=road_width_ft,
                        )
                    mark("path_plan_ms")

                    path_ok = lane_local is not None and len(lane_local) >= 4
                    if path_ok:
                        steer_est.update_bev(lane_local)
                        latest_path = [
                            [float(x), float(y)] for x, y in lane_local[:: max(1, len(lane_local) // 40)]
                        ]
                    else:
                        latest_path = []
                    # Keep autosteer "fresh" even when the planner fails — we
                    # hold the last commanded steering so the wheel doesn't
                    # toggle stale on momentary centerline dropouts.
                    inference_ok = True
                    mark("path_state_ms")

                    latest_ego_speed_mph, latest_ego_speed_ok = read_ego_speed_mph(
                        args.ego_state_file
                    )
                    latest_lookahead_m = adaptive_lookahead_m(latest_ego_speed_mph, latest_ego_speed_ok)
                    if launch_start_t is None:
                        launch_start_t = time.monotonic()
                    ramp_age = time.monotonic() - launch_start_t
                    # Always ramp toward at least stiction-break gas so low
                    # mph targets (ff < stiction) still get the kart rolling.
                    ramp_top = max(target_gas_ff, STICTION_GAS_BREAK)
                    if LAUNCH_RAMP_S > 0.0 and ramp_age < LAUNCH_RAMP_S:
                        frac = max(0.0, ramp_age / LAUNCH_RAMP_S)
                        gas_ceiling = LAUNCH_GAS_MIN + frac * max(
                            0.0, ramp_top - LAUNCH_GAS_MIN
                        )
                    else:
                        gas_ceiling = 1.0
                    latest_target_gas, latest_gas_trim, latest_target_brake = speed_ctrl.step(
                        args.target_mph,
                        latest_ego_speed_mph,
                        latest_ego_speed_ok,
                        target_gas_ff,
                        gas_ceiling=gas_ceiling,
                    )
                    # Stuck-detector: if we're still essentially parked after
                    # the launch ramp ends, override the controller with
                    # stiction-break gas to get the wheels rolling. Once ego
                    # > STICTION_EGO_MPH the PI takes over and pulls back.
                    now_mono = time.monotonic()
                    if (latest_ego_speed_ok
                            and ramp_age > LAUNCH_RAMP_S
                            and args.target_mph > STICTION_EGO_MPH
                            and latest_ego_speed_mph < STICTION_EGO_MPH):
                        if stuck_since is None:
                            stuck_since = now_mono
                        elif now_mono - stuck_since > STICTION_STUCK_S:
                            latest_target_gas = max(latest_target_gas, STICTION_GAS_BREAK)
                            latest_target_brake = 0.0
                    else:
                        stuck_since = None
                    if path_ok:
                        heading_steer = lookahead_heading_steering_deg(lane_local, latest_lookahead_m)
                        if heading_steer is not None:
                            latest_steer_base = heading_steer * STEERING_SIGN
                        else:
                            latest_steer_base = float(steer_est.steering_deg) * STEERING_SIGN
                        latest_steer_filtered = (
                            STEERING_EMA * latest_steer_base
                            + (1.0 - STEERING_EMA) * latest_steer_filtered
                        )
                        latest_steer_raw = float(np.clip(latest_steer_filtered, -270.0, 270.0))
                    # else: no path — hold latest_steer_* at their previous values.
                    mark("steer_ms")
                    overlay_rgb = create_overlay(frame_rgb, seg_map)
                    latest_overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    mark("overlay_ms")
                    latest_bev_bgr = draw_bev_viz(bev_rgb, lane_traj, lane_local, rt)
                    mark("bev_viz_ms")

                    infer_times.append(time.perf_counter() - t0)
                    if len(infer_times) > 20:
                        infer_times.pop(0)
                    infer_count += 1
                    stage_ms["infer_total_ms"] = (time.perf_counter() - stage_start) * 1000.0
                    latest_latency_ms = {k: round(float(v), 3) for k, v in stage_ms.items()}
                    if args.profile_every > 0 and infer_count % args.profile_every == 0:
                        ordered = " ".join(
                            f"{k}={latest_latency_ms[k]:.1f}"
                            for k in sorted(latest_latency_ms)
                        )
                        print(f"[profile] frame={infer_count} {ordered}", flush=True)
                next_infer_t = now + infer_period

            if now >= next_publish_t:
                publish_start = time.perf_counter()
                jpeg_ms: dict[str, float] = {}
                for r in readers:
                    f = r.latest()
                    if f is not None:
                        t = time.perf_counter()
                        write_jpeg_atomic(args.frames_dir / f"{r.slug}.jpg", f)
                        jpeg_ms[f"jpeg_{r.slug}_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_overlay_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "seg.jpg", latest_overlay_bgr)
                    jpeg_ms["jpeg_seg_ms"] = (time.perf_counter() - t) * 1000.0
                if latest_bev_bgr is not None:
                    t = time.perf_counter()
                    write_jpeg_atomic(args.frames_dir / "bev.jpg", latest_bev_bgr, quality=85)
                    jpeg_ms["jpeg_bev_ms"] = (time.perf_counter() - t) * 1000.0
                publish_jpeg_total_ms = (time.perf_counter() - publish_start) * 1000.0

                mean_dt = sum(infer_times) / len(infer_times) if infer_times else 0.0
                fps = 1.0 / mean_dt if mean_dt > 0 else 0.0
                state = {
                    "steer_deg": latest_steer_raw,
                    "steer_deg_raw": latest_steer_raw,
                    "active_cam": ACTIVE_SLUG,
                    "inference": inference_ok,
                    "viz": latest_overlay_bgr is not None or latest_bev_bgr is not None,
                    "viz_streams": [
                        slug for slug, img in (("seg", latest_overlay_bgr), ("bev", latest_bev_bgr))
                        if img is not None
                    ],
                    "object_count": 0,
                    "fps": float(fps),
                    "cams": [r.slug for r in readers],
                    "source": "video" if args.video else "camera",
                    "model": MODEL_NAME,
                    "model_full": f"drive-by-segmentation-segformer-{args.model}",
                    "target_speed_mph": args.target_mph if inference_ok else 0.0,
                    "ego_speed_mph": latest_ego_speed_mph,
                    "ego_speed_ok": latest_ego_speed_ok,
                    "steer_deg_base": latest_steer_base,
                    "steer_lookahead_m": latest_lookahead_m,
                    "target_gas": latest_target_gas if inference_ok else 0.0,
                    "target_gas_ff": target_gas_ff,
                    "target_gas_trim": latest_gas_trim,
                    "speed_mode": "arkit_pi" if latest_ego_speed_ok else "feedforward_pot",
                    "target_brake": latest_target_brake if inference_ok else 0.0,
                    "predicted_path": latest_path if inference_ok else [],
                    "segmentation": {
                        "bev_mode": args.bev_mode,
                        "latency_ms": latest_latency_ms,
                        "jpeg_ms": {k: round(float(v), 3) for k, v in jpeg_ms.items()},
                        "publish_jpeg_total_ms": round(float(publish_jpeg_total_ms), 3),
                        "infer_count": int(infer_count),
                    },
                    "ts": time.time(),
                }
                t = time.perf_counter()
                write_json_atomic(args.state_file, state)
                latest_latency_ms["json_state_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
                next_publish_t = now + publish_period

            time.sleep(0.005)
    finally:
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()
