"""Plan and render lane-center trajectories on BEV segmentation maps.

Uses SegFormer seg_maps.npz (from cloud/eval_segmentation.py) + map route +
GPS + control log. The map route (A→B) provides the intended path; GPS
localizes the cart on it; the planner predicts a driveable trajectory, and
the predicted steering is drawn next to ground truth.

    python tools/render_trajectories.py      # or: modal run cloud/render_trajectories.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive import draw
from segdrive.bev import create_bev, create_overlay, road_mask_from_bev
from segdrive.draw import (
    BEV_SIZE, draw_gps_position, draw_map_route, draw_steering_wheel, draw_trajectory, lookahead_point,
)
from segdrive.paths import DATA_DIR, EVAL_DIR, load_calib
from segdrive.planning import (
    get_ego_from_gps, get_map_route_ahead, lane_aware_centerline_path, load_ego_data,
    load_gps_route, load_map_route, localize_on_map_route, map_route_to_bev,
)
from segdrive.steering import SteeringEstimator, get_gt_steering_at_time, load_control_data
from segdrive.video import encode_h264


def main():
    data_dir = str(DATA_DIR)
    video_path = os.path.join(data_dir, "front-wide.mp4")
    gps_path = os.path.join(data_dir, "gps.json")
    ego_path = os.path.join(data_dir, "ego.jsonl")
    seg_dir = str(EVAL_DIR / "segformer")
    output_dir = str(EVAL_DIR / "trajectories")
    os.makedirs(output_dir, exist_ok=True)

    calib = load_calib()
    RANGE_FWD, RANGE_SIDE = draw.set_bev_range(calib)

    control_path = os.path.join(data_dir, "control.jsonl")
    map_route_path = os.path.join(data_dir, "map_route.json")

    gps_route, ref_lat, ref_lon = load_gps_route(gps_path)
    map_route = load_map_route(map_route_path, ref_lat, ref_lon)
    _ = load_ego_data(ego_path)  # available if needed
    controls = load_control_data(control_path)
    print(f"Loaded map route: {len(map_route)} waypoints")

    seg_data = np.load(os.path.join(seg_dir, "seg_maps.npz"))
    seg_maps = seg_data["seg_maps"]
    frame_indices = seg_data["frame_indices"]

    cap = cv2.VideoCapture(video_path)
    fps_video = 30

    combined_frames = []
    overlay_frames = []
    pred_steers = []
    gt_steers = []

    total = len(frame_indices)
    max_time = frame_indices[-1] / fps_video if total > 0 else 0
    print(f"Planning trajectories for {total} frames (SegFormer, {max_time:.0f}s)...")

    # Causal steering estimator — vision-only, processes frames one at a time
    steer_est = SteeringEstimator()

    smoothed_gt = 0.0

    for i, frame_idx in enumerate(frame_indices):
        t = frame_idx / fps_video
        seg_map = seg_maps[i]

        ego_x, ego_y, ego_yaw, ego_speed = get_ego_from_gps(gps_route, t)

        map_idx, map_dist = localize_on_map_route(map_route, ego_x, ego_y)
        map_ahead = get_map_route_ahead(map_route, map_idx)

        # Render the BEV — this is the exact image shown in the video
        bev_base = create_bev(seg_map, calib, BEV_SIZE)

        # Extract road mask directly from the rendered BEV pixel colors
        road_mask = road_mask_from_bev(bev_base)

        mr_bx, mr_by = map_route_to_bev(map_route, ego_x, ego_y, ego_yaw, BEV_SIZE, RANGE_FWD, RANGE_SIDE)

        # Lane-aware centerline planner
        t0 = time.time()
        road_width_ft = calib.get("road_width_ft", 20.0)
        lane_traj, lane_local = lane_aware_centerline_path(
            road_mask, bev_size=BEV_SIZE, range_fwd=RANGE_FWD, range_side=RANGE_SIDE,
            road_mask=road_mask, gps_bx=mr_bx, gps_by=mr_by,
            road_width_ft=road_width_ft
        )
        lane_ms = (time.time() - t0) * 1000

        # Feed BEV path into steering estimator for anticipatory correction
        steer_est.update_bev(lane_local)
        pred_steers.append(steer_est.steering_deg)

        raw_gt = get_gt_steering_at_time(controls, t)
        smoothed_gt = 0.3 * raw_gt + 0.7 * smoothed_gt
        gt_steers.append(smoothed_gt)

        la_point, la_ft = lookahead_point(lane_local)

        # ── BEV + trajectory + lookahead ──
        bev_frame = bev_base.copy()
        draw_map_route(bev_frame, map_route, ego_x, ego_y, ego_yaw)
        draw_trajectory(bev_frame, lane_traj, (255, 255, 0), 3, "Lane Center")

        # Draw lookahead point and line from ego
        if la_point is not None:
            la_bx = int((la_point[1] / RANGE_SIDE * 0.5 + 0.5) * BEV_SIZE)
            la_by = int((1 - la_point[0] / RANGE_FWD) * BEV_SIZE)
            ego_bx, ego_by = BEV_SIZE // 2, BEV_SIZE - 1
            if 0 <= la_bx < BEV_SIZE and 0 <= la_by < BEV_SIZE:
                cv2.line(bev_frame, (ego_bx, ego_by), (la_bx, la_by),
                         (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(bev_frame, (la_bx, la_by), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(bev_frame, (la_bx, la_by), 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(bev_frame, f"LA {la_ft:.0f}ft",
                            (la_bx + 10, la_by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        draw_gps_position(bev_frame, ego_x, ego_y, ego_yaw)
        combined_frames.append(cv2.cvtColor(bev_frame, cv2.COLOR_RGB2BGR))

        # Overlay camera frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = create_overlay(frame_rgb, seg_map)
            overlay_frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            overlay_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Lane: {lane_ms:.0f}ms"
                  f"  speed={ego_speed:.1f}m/s  map_idx={map_idx}/{len(map_route)}"
                  f"  map_dist={map_dist:.1f}m")

    cap.release()

    # Compute per-frame repeat counts to fill 30fps output
    repeats = []
    for idx in range(len(frame_indices)):
        if idx + 1 < len(frame_indices):
            repeats.append(int(frame_indices[idx + 1] - frame_indices[idx]))
        else:
            repeats.append(repeats[-1] if repeats else 15)

    # Encode videos
    print("\nEncoding videos...")

    def expand_30fps(frames):
        out = []
        for i, f in enumerate(frames):
            r = repeats[i] if i < len(repeats) else 15
            for _ in range(r):
                out.append(f)
        return out

    encode_h264(expand_30fps(combined_frames), os.path.join(output_dir, "bev_lane_center.mp4"), fps=30)

    if overlay_frames:
        encode_h264(expand_30fps(overlay_frames), os.path.join(output_dir, "overlay.mp4"), fps=30)

    # Side-by-side: overlay + BEV + steering wheels
    sbs_frames = []
    wheel_radius = 70
    wheel_panel_h = wheel_radius * 2 + 60
    h_target = 480
    for i in range(min(len(overlay_frames), len(combined_frames))):
        ov = overlay_frames[i]
        cb = combined_frames[i]

        ov_resized = cv2.resize(ov, (640, h_target))
        cb_resized = cv2.resize(cb, (h_target, h_target))

        total_w = 640 + h_target
        canvas = np.zeros((h_target + wheel_panel_h, total_w, 3), dtype=np.uint8)
        canvas[:h_target, :640] = ov_resized
        canvas[:h_target, 640:] = cb_resized

        cv2.putText(canvas, "Segmentation Overlay", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "BEV + Trajectory", (650, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Legend
        lx = 650
        ly = 50
        legend_items = [
            ("Map Route (A->B)", (0, 220, 255)),
            ("Lane Center Pred", (255, 255, 0)),
            ("Lookahead (speed-scaled)", (0, 255, 255)),
            ("GPS Position", (50, 150, 255)),
        ]
        for label, color in legend_items:
            cv2.circle(canvas, (lx, ly), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (lx + 12, ly + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            ly += 18

        # Divider line
        cv2.line(canvas, (0, h_target), (total_w, h_target), (60, 60, 60), 1)

        # Steering wheels
        wheel_y = h_target + wheel_radius + 12
        pred_wx = total_w // 4
        gt_wx = 3 * total_w // 4

        pred_col_deg = pred_steers[i] if i < len(pred_steers) else 0
        gt_col_deg = gt_steers[i] if i < len(gt_steers) else 0

        draw_steering_wheel(canvas, pred_wx, wheel_y, wheel_radius,
                           pred_col_deg, "Predicted", (255, 255, 0))
        draw_steering_wheel(canvas, gt_wx, wheel_y, wheel_radius,
                           gt_col_deg, "Ground Truth", (255, 255, 255))

        sbs_frames.append(canvas)

    encode_h264(expand_30fps(sbs_frames), os.path.join(output_dir, "side_by_side.mp4"), fps=30)

    # Save sample images
    for j, si in enumerate([0, total // 4, total // 2, 3 * total // 4]):
        if si < len(combined_frames):
            cv2.imwrite(os.path.join(output_dir, f"sample_{j}_combined.jpg"),
                       combined_frames[si], [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  bev_lane_center.mp4, side_by_side.mp4")
    print(f"  {min(4, len(combined_frames))} sample images")


if __name__ == "__main__":
    main()
