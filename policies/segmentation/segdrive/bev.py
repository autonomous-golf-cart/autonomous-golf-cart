"""Bird's-eye-view projection of a segmentation map through the calibrated
fisheye camera model, plus the camera-view color overlay."""
import math

import cv2
import numpy as np

from .cityscapes import CITYSCAPES_COLORS, ROAD


def create_overlay(frame_rgb, seg_map, alpha=0.45):
    color_mask = np.zeros_like(frame_rgb)
    for cls_id, color in enumerate(CITYSCAPES_COLORS):
        mask = seg_map == cls_id
        color_mask[mask] = color
    return ((1 - alpha) * frame_rgb + alpha * color_mask).astype(np.uint8)


def create_bev(seg_map, calib, bev_size=500, return_class_map=False):
    f = calib["intrinsics"]["focal_length"]
    cx_param = calib["intrinsics"]["cx"]
    cy_param = calib["intrinsics"]["cy"]
    k1 = calib["intrinsics"]["k1"]
    k2 = calib["intrinsics"]["k2"]
    h = calib["extrinsics"]["height_m"]
    img_h, img_w = seg_map.shape[:2]

    FT_TO_M = 0.3048
    bev_range = calib.get("bev_range", {})
    range_forward_ft = bev_range.get("forward_ft", 50)
    range_side_ft = bev_range.get("side_ft", 25)
    range_forward = range_forward_ft * FT_TO_M
    range_side = range_side_ft * FT_TO_M

    pitch = math.radians(calib["extrinsics"]["pitch_deg"])
    roll = math.radians(calib["extrinsics"]["roll_deg"])
    yaw = math.radians(calib["extrinsics"]["yaw_deg"])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cyw, syw = math.cos(yaw), math.sin(yaw)

    def matmul(A, B):
        C = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for kk in range(3):
                    C[i][j] += A[i][kk] * B[kk][j]
        return C

    Ryaw = [[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]]
    Rbase = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    Rpitch = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    Rroll = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
    R = matmul(Rroll, matmul(Rpitch, matmul(Rbase, Ryaw)))

    bev_color = np.full((bev_size, bev_size, 3), 30, dtype=np.uint8)
    in_fov = np.zeros((bev_size, bev_size), dtype=bool)

    by_arr, bx_arr = np.mgrid[0:bev_size, 0:bev_size]
    by_flat = by_arr.ravel()
    bx_flat = bx_arr.ravel()

    wx = (bx_flat / bev_size - 0.5) * 2 * range_side
    wy = (1 - by_flat / bev_size) * range_forward

    mask = wy > 0.15
    wx, wy = wx[mask], wy[mask]
    by_f, bx_f = by_flat[mask], bx_flat[mask]

    dx, dy, dz = wx, wy, np.full_like(wx, -h)

    cam_x = R[0][0]*dx + R[0][1]*dy + R[0][2]*dz
    cam_y = R[1][0]*dx + R[1][1]*dy + R[1][2]*dz
    cam_z = R[2][0]*dx + R[2][1]*dy + R[2][2]*dz

    m2 = cam_z > 0.01
    cam_x, cam_y, cam_z = cam_x[m2], cam_y[m2], cam_z[m2]
    by_f, bx_f = by_f[m2], bx_f[m2]

    r3d = np.sqrt(cam_x**2 + cam_y**2)
    theta = np.arctan2(r3d, cam_z)
    m3 = theta < math.pi * 0.47
    cam_x, cam_y, r3d, theta = cam_x[m3], cam_y[m3], r3d[m3], theta[m3]
    by_f, bx_f = by_f[m3], bx_f[m3]

    t2 = theta**2
    td = theta * (1 + k1*t2 + k2*t2*t2)
    rp = f * td

    safe = r3d > 1e-8
    u = np.where(safe, cx_param + rp * cam_x / r3d, cx_param)
    v = np.where(safe, cy_param + rp * cam_y / r3d, cy_param)

    iu = np.floor(u).astype(np.int32)
    iv = np.floor(v).astype(np.int32)
    m4 = (iu >= 0) & (iu < img_w) & (iv >= 0) & (iv < img_h)
    iu, iv = iu[m4], iv[m4]
    by_f, bx_f = by_f[m4], bx_f[m4]

    in_fov[by_f, bx_f] = True
    cls_ids = seg_map[iv, iu]

    # Class-ID map: same projection as the color image, before any overlays
    if return_class_map:
        cls_map = np.full((bev_size, bev_size), 255, dtype=np.uint8)
        cls_map[by_f, bx_f] = cls_ids

    colors = np.array(CITYSCAPES_COLORS, dtype=np.uint8)
    valid_cls = cls_ids < len(CITYSCAPES_COLORS)
    bev_color[by_f[valid_cls], bx_f[valid_cls]] = colors[cls_ids[valid_cls]]

    # FOV boundary
    fov_border = np.zeros((bev_size, bev_size), dtype=np.uint8)
    fov_border[in_fov] = 255
    dilated = cv2.dilate(fov_border, np.ones((3, 3), dtype=np.uint8))
    bev_color[(dilated > 0) & (~in_fov)] = (80, 80, 80)

    # Grid lines inside FOV
    for dist_ft in range(10, range_forward_ft + 1, 10):
        by_grid = int((1 - dist_ft * FT_TO_M / range_forward) * bev_size)
        if 0 <= by_grid < bev_size:
            row_fov = in_fov[by_grid, :]
            row = bev_color[by_grid].astype(np.int16)
            row[row_fov] = np.clip(row[row_fov] + 35, 0, 255)
            bev_color[by_grid] = row.astype(np.uint8)

    for dist_ft in range(-20, range_side_ft + 1, 10):
        bx_grid = int((dist_ft / range_side_ft * 0.5 + 0.5) * bev_size)
        if 0 <= bx_grid < bev_size:
            col_fov = in_fov[:, bx_grid]
            col = bev_color[:, bx_grid].astype(np.int16)
            col[col_fov] = np.clip(col[col_fov] + 35, 0, 255)
            bev_color[:, bx_grid] = col.astype(np.uint8)

    # Distance labels
    for dist_ft in range(10, range_forward_ft + 1, 10):
        by_grid = int((1 - dist_ft * FT_TO_M / range_forward) * bev_size)
        if 0 <= by_grid < bev_size:
            cv2.putText(bev_color, f"{dist_ft}ft", (4, by_grid - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    # Ego car triangle
    ex, ey = bev_size // 2, bev_size - 8
    pts = np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]])
    cv2.fillPoly(bev_color, [pts], (255, 255, 255))

    if return_class_map:
        return bev_color, cls_map
    return bev_color



def road_mask_from_bev(bev_rgb):
    """Boolean mask of road pixels in a rendered BEV image.

    Grid lines brighten road pixels by +35 (and +70 where two lines cross), so
    those shades count as road too.
    """
    road = np.array(CITYSCAPES_COLORS[ROAD], dtype=np.int16)
    mask = np.zeros(bev_rgb.shape[:2], dtype=bool)
    for boost in (0, 35, 70):
        color = np.clip(road + boost, 0, 255).astype(np.uint8)
        mask |= np.all(bev_rgb == color, axis=-1)
    return mask
