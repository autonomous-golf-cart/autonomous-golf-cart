"""Drawing helpers for BEV frames: trajectories, routes, lookahead, steering wheel.

BEV geometry lives in module globals so every helper agrees on scale. Call
``set_bev_range(calib)`` once after loading the calibration to match its
``bev_range``.
"""
import math

import cv2
import numpy as np

FT_TO_M = 0.3048
RANGE_FWD = 50 * FT_TO_M
RANGE_SIDE = 25 * FT_TO_M
BEV_SIZE = 500

LOOKAHEAD_FT = 25.0
LOOKAHEAD_M = LOOKAHEAD_FT * FT_TO_M


def set_bev_range(calib):
    """Set RANGE_FWD / RANGE_SIDE from the calibration's ``bev_range`` (feet)."""
    global RANGE_FWD, RANGE_SIDE
    bev_range = calib.get("bev_range", {})
    RANGE_FWD = bev_range.get("forward_ft", 50) * FT_TO_M
    RANGE_SIDE = bev_range.get("side_ft", 25) * FT_TO_M
    return RANGE_FWD, RANGE_SIDE


def lookahead_point(traj_local):
    """BEV path lookahead point for visualization."""
    if traj_local is None or len(traj_local) < 4:
        return None, 0.0

    fwd = traj_local[1:, 0]
    left = traj_local[1:, 1]

    if len(fwd) < 5:
        return None, 0.0

    coeffs = np.polyfit(fwd, left, 2)
    la_fwd_val = LOOKAHEAD_M
    la_left_val = float(np.polyval(coeffs, la_fwd_val))

    return (la_fwd_val, la_left_val), LOOKAHEAD_FT


def draw_steering_wheel(img, cx, cy, radius, angle_deg, label, color, bg_color=(30, 30, 30)):
    """Draw a steering wheel rotated by angle_deg with label underneath."""
    size = radius * 2 + 40
    wheel_img = np.full((size, size, 3), bg_color[0], dtype=np.uint8)
    wcx, wcy = size // 2, size // 2

    angle_rad = math.radians(angle_deg)

    # Outer ring
    cv2.circle(wheel_img, (wcx, wcy), radius, color, 3, cv2.LINE_AA)
    # Inner ring
    cv2.circle(wheel_img, (wcx, wcy), radius - 12, color, 2, cv2.LINE_AA)

    # Three spokes at 120° apart, rotated by steering angle
    for spoke_angle in [0, 120, 240]:
        a = angle_rad + math.radians(spoke_angle)
        sx = int(wcx + (radius - 12) * math.cos(a))
        sy = int(wcy + (radius - 12) * math.sin(a))
        ix = int(wcx + 10 * math.cos(a))
        iy = int(wcy + 10 * math.sin(a))
        cv2.line(wheel_img, (ix, iy), (sx, sy), color, 2, cv2.LINE_AA)

    # Center hub
    cv2.circle(wheel_img, (wcx, wcy), 8, color, -1, cv2.LINE_AA)

    # Top marker (reference notch) — always at 12 o'clock
    cv2.circle(wheel_img, (wcx, wcy - radius + 6), 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Rotated position marker — shows where the top marker has moved
    marker_x = int(wcx + (radius - 6) * math.sin(angle_rad))
    marker_y = int(wcy - (radius - 6) * math.cos(angle_rad))
    cv2.circle(wheel_img, (marker_x, marker_y), 5, color, -1, cv2.LINE_AA)

    # Place onto target image
    x1 = cx - size // 2
    y1 = cy - size // 2
    h, w = img.shape[:2]
    # Clip to image bounds
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    sx2 = min(size, w - x1)
    sy2 = min(size, h - y1)
    if sx2 > sx1 and sy2 > sy1:
        img[dy1:dy1 + sy2 - sy1, dx1:dx1 + sx2 - sx1] = wheel_img[sy1:sy2, sx1:sx2]

    # Label underneath
    text = f"{label}: {angle_deg:+.1f} deg"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    tx = cx - text_size[0] // 2
    ty = cy + radius + 28
    if 0 <= tx < w and 0 <= ty < h:
        cv2.putText(img, text, (tx, ty),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_trajectory(bev_img, traj_bev, color, thickness=3, label=None):
    """Draw a trajectory line on a BEV image."""
    if traj_bev is None or len(traj_bev) < 2:
        return
    pts = traj_bev.copy()
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < BEV_SIZE) & (pts[:, 1] >= 0) & (pts[:, 1] < BEV_SIZE)
    if valid.sum() < 2:
        return

    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(bev_img,
                     (int(pts[i, 0]), int(pts[i, 1])),
                     (int(pts[i + 1, 0]), int(pts[i + 1, 1])),
                     color, thickness, cv2.LINE_AA)

    for i in range(0, len(pts), 5):
        if valid[i]:
            cv2.circle(bev_img, (int(pts[i, 0]), int(pts[i, 1])), 3, color, -1, cv2.LINE_AA)

    if label:
        for i in range(len(pts)):
            if valid[i]:
                cv2.putText(bev_img, label, (int(pts[i, 0]) + 6, int(pts[i, 1]) - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                break


def draw_gps_route(bev_img, gps_route, ego_x, ego_y, ego_yaw, ref_lat, ref_lon):
    """Draw the raw GPS recording on BEV as a thin dotted line."""
    wx = gps_route[:, 1]
    wy = gps_route[:, 2]

    dx = wx - ego_x
    dy = wy - ego_y
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c

    bx = ((local_left / RANGE_SIDE * 0.5 + 0.5) * BEV_SIZE).astype(int)
    by = ((1 - local_fwd / RANGE_FWD) * BEV_SIZE).astype(int)

    valid = (bx >= 0) & (bx < BEV_SIZE) & (by >= 0) & (by < BEV_SIZE)
    valid &= (local_fwd > 0) & (local_fwd < RANGE_FWD)

    for i in range(len(bx) - 1):
        if valid[i] and valid[i + 1] and i % 3 == 0:
            cv2.line(bev_img, (bx[i], by[i]), (bx[i + 1], by[i + 1]),
                     (200, 200, 200), 1, cv2.LINE_AA)


def draw_map_route(bev_img, map_route, ego_x, ego_y, ego_yaw):
    """Draw the planned map route (A→B) on BEV as a thick cyan dashed line."""
    dx = map_route[:, 0] - ego_x
    dy = map_route[:, 1] - ego_y
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    local_fwd = dx * c - dy * s
    local_left = dx * s + dy * c

    bx = ((local_left / RANGE_SIDE * 0.5 + 0.5) * BEV_SIZE).astype(int)
    by = ((1 - local_fwd / RANGE_FWD) * BEV_SIZE).astype(int)

    valid = (bx >= 0) & (bx < BEV_SIZE) & (by >= 0) & (by < BEV_SIZE)
    valid &= (local_fwd > 0) & (local_fwd < RANGE_FWD)

    color = (0, 220, 255)  # cyan/yellow
    for i in range(len(bx) - 1):
        if valid[i] and valid[i + 1]:
            if (i // 2) % 2 == 0:
                cv2.line(bev_img, (bx[i], by[i]), (bx[i + 1], by[i + 1]),
                         color, 2, cv2.LINE_AA)

    for i in range(0, len(bx), 3):
        if valid[i]:
            cv2.circle(bev_img, (bx[i], by[i]), 4, color, -1, cv2.LINE_AA)

    # Label
    for i in range(len(bx)):
        if valid[i]:
            cv2.putText(bev_img, "Map Route", (int(bx[i]) + 8, int(by[i]) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            break


def draw_gps_position(bev_img, ego_x, ego_y, ego_yaw):
    """Draw the current GPS position as a prominent dot at BEV ego origin."""
    bx = BEV_SIZE // 2
    by = BEV_SIZE - 1
    color = (50, 150, 255)  # orange
    cv2.circle(bev_img, (bx, by), 8, color, -1, cv2.LINE_AA)
    cv2.circle(bev_img, (bx, by), 10, (255, 255, 255), 1, cv2.LINE_AA)
    # Heading arrow
    arrow_len = 20
    ax = int(bx + arrow_len * math.sin(0))
    ay = int(by - arrow_len * math.cos(0))
    cv2.arrowedLine(bev_img, (bx, by), (ax, ay), color, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(bev_img, "GPS", (bx + 14, by + 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
