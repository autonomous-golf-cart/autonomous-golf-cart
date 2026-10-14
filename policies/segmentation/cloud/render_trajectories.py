"""Run tools/render_trajectories.py on Modal for fast CPU rendering.

    modal run cloud/render_trajectories.py
"""
import modal
import sys
import os
import json

app = modal.App("drive-by-render-trajectories")

render_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "scipy",
    )
)

vol = modal.Volume.from_name("segmentation-data", create_if_missing=True)


@app.function(
    image=render_image,
    cpu=8,
    memory=16384,
    timeout=1800,
)
def run_render(
    seg_npz_bytes: bytes,
    video_bytes: bytes,
    calib_json: str,
    gps_json: str,
    control_jsonl: str,
    map_route_json: str,
    segdrive_src: dict,
):
    import numpy as np
    import cv2
    import math
    import time
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp()
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir)

    with open(os.path.join(data_dir, "front-wide.mp4"), "wb") as f:
        f.write(video_bytes)
    with open(os.path.join(tmpdir, "camera_calibration.json"), "w") as f:
        f.write(calib_json)
    with open(os.path.join(data_dir, "gps.json"), "w") as f:
        f.write(gps_json)
    with open(os.path.join(data_dir, "control.jsonl"), "w") as f:
        f.write(control_jsonl)
    with open(os.path.join(data_dir, "map_route.json"), "w") as f:
        f.write(map_route_json)

    seg_path = os.path.join(tmpdir, "eval_output", "segformer")
    os.makedirs(seg_path)
    with open(os.path.join(seg_path, "seg_maps.npz"), "wb") as f:
        f.write(seg_npz_bytes)

    output_dir = os.path.join(tmpdir, "eval_output", "trajectories")
    os.makedirs(output_dir, exist_ok=True)

    import sys

    # Recreate the segdrive package from the shipped sources
    pkg_dir = os.path.join(tmpdir, "segdrive")
    os.makedirs(pkg_dir)
    for name, src in segdrive_src.items():
        with open(os.path.join(pkg_dir, name), "w") as f:
            f.write(src)

    sys.path.insert(0, tmpdir)
    os.chdir(tmpdir)

    from segdrive import draw
    from segdrive.bev import create_bev, create_overlay, road_mask_from_bev
    from segdrive.draw import (
        lookahead_point, draw_trajectory, draw_map_route, draw_gps_position, draw_steering_wheel,
    )
    from segdrive.planning import (
        load_gps_route, get_ego_from_gps,
        lane_aware_centerline_path, map_route_to_bev,
        load_map_route, localize_on_map_route,
    )
    from segdrive.steering import SteeringEstimator, load_control_data, get_gt_steering_at_time
    from segdrive.video import encode_h264

    calib = json.loads(calib_json)
    RANGE_FWD, RANGE_SIDE = draw.set_bev_range(calib)
    BEV_SIZE = draw.BEV_SIZE

    gps_route, ref_lat, ref_lon = load_gps_route(os.path.join(data_dir, "gps.json"))
    map_route = load_map_route(os.path.join(data_dir, "map_route.json"), ref_lat, ref_lon)
    controls = load_control_data(os.path.join(data_dir, "control.jsonl"))

    seg_data = np.load(os.path.join(seg_path, "seg_maps.npz"))
    seg_maps = seg_data["seg_maps"]
    frame_indices = seg_data["frame_indices"]

    cap = cv2.VideoCapture(os.path.join(data_dir, "front-wide.mp4"))
    fps_video = 30

    steer_est = SteeringEstimator()
    smoothed_gt = 0.0

    combined_frames = []
    overlay_frames = []
    pred_steers = []
    gt_steers = []

    total = len(frame_indices)
    max_time = frame_indices[-1] / fps_video if total > 0 else 0
    print(f"Rendering {total} frames ({max_time:.0f}s)...")

    road_width_ft = calib.get("road_width_ft", 20.0)

    t_start = time.time()
    for i, frame_idx in enumerate(frame_indices):
        t = frame_idx / fps_video
        seg_map = seg_maps[i]
        ego_x, ego_y, ego_yaw, ego_speed = get_ego_from_gps(gps_route, t)

        map_idx, map_dist = localize_on_map_route(map_route, ego_x, ego_y)

        bev_base = create_bev(seg_map, calib, BEV_SIZE)
        road_mask = road_mask_from_bev(bev_base)
        mr_bx, mr_by = map_route_to_bev(map_route, ego_x, ego_y, ego_yaw, BEV_SIZE, RANGE_FWD, RANGE_SIDE)

        lane_traj, lane_local = lane_aware_centerline_path(
            road_mask, bev_size=BEV_SIZE, range_fwd=RANGE_FWD, range_side=RANGE_SIDE,
            road_mask=road_mask, gps_bx=mr_bx, gps_by=mr_by, road_width_ft=road_width_ft
        )

        steer_est.update_bev(lane_local)
        pred_steers.append(steer_est.steering_deg)

        raw_gt = get_gt_steering_at_time(controls, t)
        smoothed_gt = 0.3 * raw_gt + 0.7 * smoothed_gt
        gt_steers.append(smoothed_gt)

        la_point, la_ft = lookahead_point(lane_local)

        bev_frame = bev_base.copy()
        draw_map_route(bev_frame, map_route, ego_x, ego_y, ego_yaw)
        draw_trajectory(bev_frame, lane_traj, (255, 255, 0), 3, "Lane Center")

        if la_point is not None:
            la_bx = int((la_point[1] / RANGE_SIDE * 0.5 + 0.5) * BEV_SIZE)
            la_by = int((1 - la_point[0] / RANGE_FWD) * BEV_SIZE)
            ego_bx, ego_by = BEV_SIZE // 2, BEV_SIZE - 1
            if 0 <= la_bx < BEV_SIZE and 0 <= la_by < BEV_SIZE:
                cv2.line(bev_frame, (ego_bx, ego_by), (la_bx, la_by), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(bev_frame, (la_bx, la_by), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(bev_frame, (la_bx, la_by), 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(bev_frame, f"LA {la_ft:.0f}ft", (la_bx + 10, la_by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        draw_gps_position(bev_frame, ego_x, ego_y, ego_yaw)
        combined_frames.append(cv2.cvtColor(bev_frame, cv2.COLOR_RGB2BGR))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = create_overlay(frame_rgb, seg_map)
            overlay_frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            overlay_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (total - i - 1) / fps
            print(f"  [{i+1}/{total}] {fps:.1f} fps, ETA {eta:.0f}s")

    cap.release()
    render_time = time.time() - t_start
    print(f"Rendering done in {render_time:.1f}s ({total/render_time:.1f} fps)")

    # Frame expansion for 30fps output
    repeats = []
    for idx in range(len(frame_indices)):
        if idx + 1 < len(frame_indices):
            repeats.append(int(frame_indices[idx + 1] - frame_indices[idx]))
        else:
            repeats.append(repeats[-1] if repeats else 15)

    def expand_30fps(frames):
        out = []
        for i, f in enumerate(frames):
            r = repeats[i] if i < len(repeats) else 15
            for _ in range(r):
                out.append(f)
        return out

    # Side-by-side with steering wheels
    print("Building side-by-side frames...")
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

        lx, ly = 650, 50
        for label, color in [("Map Route (A->B)", (0, 220, 255)), ("Lane Center Pred", (255, 255, 0)),
                              ("Lookahead", (0, 255, 255)), ("GPS Position", (50, 150, 255))]:
            cv2.circle(canvas, (lx, ly), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (lx + 12, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            ly += 18

        cv2.line(canvas, (0, h_target), (total_w, h_target), (60, 60, 60), 1)
        wheel_y = h_target + wheel_radius + 12
        draw_steering_wheel(canvas, total_w // 4, wheel_y, wheel_radius,
                            pred_steers[i], "Predicted", (255, 255, 0))
        draw_steering_wheel(canvas, 3 * total_w // 4, wheel_y, wheel_radius,
                            gt_steers[i], "Ground Truth", (255, 255, 255))
        sbs_frames.append(canvas)

    print("Encoding videos...")
    encode_h264(expand_30fps(combined_frames), os.path.join(output_dir, "bev_lane_center.mp4"), fps=30)
    if overlay_frames:
        encode_h264(expand_30fps(overlay_frames), os.path.join(output_dir, "overlay.mp4"), fps=30)
    encode_h264(expand_30fps(sbs_frames), os.path.join(output_dir, "side_by_side.mp4"), fps=30)

    # Read back the video files
    result_files = {}
    for name in ["bev_lane_center.mp4", "overlay.mp4", "side_by_side.mp4"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                result_files[name] = f.read()

    # Compute final stats
    p = np.array(pred_steers)
    g = np.array([get_gt_steering_at_time(controls, fi / fps_video) for fi in frame_indices])
    ae = np.abs(p - g)
    corr = np.corrcoef(p, g)[0, 1]

    return {
        "files": result_files,
        "mae": float(ae.mean()),
        "corr": float(corr),
        "render_time_s": round(render_time, 1),
        "num_frames": total,
    }


@app.local_entrypoint()
def main():
    import numpy as np

    seg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, seg_root)
    from segdrive.paths import CALIB_PATH, DATA_DIR, EVAL_DIR

    data_dir = str(DATA_DIR)
    output_dir = str(EVAL_DIR / "trajectories")
    os.makedirs(output_dir, exist_ok=True)

    print("Reading input files...")
    with open(os.path.join(data_dir, "front-wide.mp4"), "rb") as f:
        video_bytes = f.read()
    with open(CALIB_PATH) as f:
        calib_json = f.read()
    with open(os.path.join(data_dir, "gps.json")) as f:
        gps_json = f.read()
    with open(os.path.join(data_dir, "control.jsonl")) as f:
        control_jsonl = f.read()
    with open(os.path.join(data_dir, "map_route.json")) as f:
        map_route_json = f.read()
    with open(EVAL_DIR / "segformer" / "seg_maps.npz", "rb") as f:
        seg_npz_bytes = f.read()

    # Ship the segdrive package source so the container runs the same code
    pkg_dir = os.path.join(seg_root, "segdrive")
    segdrive_src = {}
    for name in sorted(os.listdir(pkg_dir)):
        if name.endswith(".py"):
            with open(os.path.join(pkg_dir, name)) as f:
                segdrive_src[name] = f.read()

    total_upload = (len(video_bytes) + len(seg_npz_bytes)) / 1e6
    print(f"Total upload: {total_upload:.0f} MB")
    print("Launching remote render...")

    result = run_render.remote(
        seg_npz_bytes=seg_npz_bytes,
        video_bytes=video_bytes,
        calib_json=calib_json,
        gps_json=gps_json,
        control_jsonl=control_jsonl,
        map_route_json=map_route_json,
        segdrive_src=segdrive_src,
    )

    print(f"\nRender complete: {result['num_frames']} frames in {result['render_time_s']}s")
    print(f"Steering MAE: {result['mae']:.1f}°  Corr: {result['corr']:.3f}")

    for name, data in result["files"].items():
        path = os.path.join(output_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {path} ({len(data)/1e6:.1f} MB)")

    print(f"\nAll videos saved to {output_dir}/")
