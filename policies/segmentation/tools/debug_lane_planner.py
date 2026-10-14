"""Run the lane-aware centerline planner on a few frames and save debug images.

Needs outputs/eval/segformer/seg_maps.npz (from cloud/eval_segmentation.py).

    python tools/debug_lane_planner.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.bev import create_bev
from segdrive.cityscapes import CITYSCAPES_COLORS
from segdrive.paths import DATA_DIR, EVAL_DIR, load_calib
from segdrive.planning import (
    load_gps_route, get_ego_from_gps,
    lane_aware_centerline_path,
    map_route_to_bev, load_map_route, localize_on_map_route, get_map_route_ahead,
)

FT_TO_M = 0.3048
RANGE_FWD = 50 * FT_TO_M
RANGE_SIDE = 25 * FT_TO_M
BEV_SIZE = 500


def draw_traj(img, traj, color, thickness=3, label=None):
    if traj is None or len(traj) < 2:
        return
    for i in range(len(traj) - 1):
        p1 = (int(traj[i, 0]), int(traj[i, 1]))
        p2 = (int(traj[i + 1, 0]), int(traj[i + 1, 1]))
        if all(0 <= c < BEV_SIZE for c in p1 + p2):
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for i in range(0, len(traj), 5):
        pt = (int(traj[i, 0]), int(traj[i, 1]))
        if 0 <= pt[0] < BEV_SIZE and 0 <= pt[1] < BEV_SIZE:
            cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)
    if label:
        for i in range(len(traj)):
            pt = (int(traj[i, 0]), int(traj[i, 1]))
            if 0 <= pt[0] < BEV_SIZE and 0 <= pt[1] < BEV_SIZE:
                cv2.putText(img, label, (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                break


def main():
    data_dir = str(DATA_DIR)
    gps_path = os.path.join(data_dir, "gps.json")
    map_route_path = os.path.join(data_dir, "map_route.json")
    seg_dir = str(EVAL_DIR / "segformer")
    out_dir = str(EVAL_DIR / "lane_debug")
    os.makedirs(out_dir, exist_ok=True)

    calib = load_calib()

    gps_route, ref_lat, ref_lon = load_gps_route(gps_path)
    map_route = load_map_route(map_route_path, ref_lat, ref_lon)
    road_width_ft = calib.get("road_width_ft", 20.0)

    seg_data = np.load(os.path.join(seg_dir, "seg_maps.npz"))
    seg_maps = seg_data["seg_maps"]
    frame_indices = seg_data["frame_indices"]

    test_frames = [36, 100, 200, 300, 400, 500, 600]

    for idx in test_frames:
        if idx >= len(frame_indices):
            continue
        frame_idx = frame_indices[idx]
        t = frame_idx / 30.0
        seg_map = seg_maps[idx]
        ego_x, ego_y, ego_yaw, ego_speed = get_ego_from_gps(gps_route, t)

        bev = create_bev(seg_map, calib, BEV_SIZE)

        road_color = np.array(CITYSCAPES_COLORS[0], dtype=np.uint8)
        grid_color = np.clip(np.array(CITYSCAPES_COLORS[0], dtype=np.int16) + 35, 0, 255).astype(np.uint8)
        grid2_color = np.clip(np.array(CITYSCAPES_COLORS[0], dtype=np.int16) + 70, 0, 255).astype(np.uint8)
        road_mask = (np.all(bev == road_color, axis=-1) |
                     np.all(bev == grid_color, axis=-1) |
                     np.all(bev == grid2_color, axis=-1))

        mr_bx, mr_by = map_route_to_bev(map_route, ego_x, ego_y, ego_yaw, BEV_SIZE, RANGE_FWD, RANGE_SIDE)

        # Lane-aware centerline
        t0 = time.time()
        la_traj, la_local = lane_aware_centerline_path(
            road_mask, BEV_SIZE, RANGE_FWD, RANGE_SIDE,
            road_mask=road_mask, gps_bx=mr_bx, gps_by=mr_by,
            road_width_ft=road_width_ft
        )
        la_ms = (time.time() - t0) * 1000

        # Draw map route
        dx = map_route[:, 0] - ego_x
        dy = map_route[:, 1] - ego_y
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        local_fwd = dx * c - dy * s
        local_left = dx * s + dy * c
        mbx = ((local_left / RANGE_SIDE * 0.5 + 0.5) * BEV_SIZE).astype(int)
        mby = ((1 - local_fwd / RANGE_FWD) * BEV_SIZE).astype(int)
        valid = (mbx >= 0) & (mbx < BEV_SIZE) & (mby >= 0) & (mby < BEV_SIZE) & (local_fwd > 0)
        for i in range(len(mbx) - 1):
            if valid[i] and valid[i + 1] and (i // 2) % 2 == 0:
                cv2.line(bev, (mbx[i], mby[i]), (mbx[i + 1], mby[i + 1]),
                         (0, 220, 255), 2, cv2.LINE_AA)

        draw_traj(bev, la_traj, (255, 255, 0), 3, "Lane Center")

        cv2.putText(bev, f"Frame {frame_idx} | Lane: {la_ms:.0f}ms",
                    (10, bev.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"debug_{idx}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(bev, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[{idx}] frame={frame_idx} lane={la_ms:.0f}ms → {out_path}")

    print(f"\nDone! Debug images in {out_dir}/")


if __name__ == "__main__":
    main()
