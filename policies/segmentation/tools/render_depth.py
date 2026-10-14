"""Client-side rendering of depth + YOLO results into side-by-side videos with 3D BEV.
Run after cloud/depth_yolo.py to generate visualization videos. No GPU needed.
Re-run anytime you change visualization params without re-running inference.
"""
import numpy as np
import cv2
import json
import os
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.paths import DATA_DIR, DEPTH_DIR, load_calib

# COCO classes relevant for self-driving obstacle avoidance (BGR colors)
OBSTACLE_COLORS = {
    0:  ("person",     (60, 60, 255)),
    1:  ("bicycle",    (0, 165, 255)),
    2:  ("car",        (255, 150, 50)),
    3:  ("motorcycle", (0, 255, 255)),
    5:  ("bus",        (255, 100, 200)),
    7:  ("truck",      (100, 220, 50)),
    9:  ("traffic light", (30, 170, 250)),
    11: ("stop sign",  (50, 50, 255)),
    15: ("cat",        (200, 150, 255)),
    16: ("dog",        (200, 150, 255)),
}

BEV_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}

MAX_DEPTH_VIS = 80.0
BEV_FORWARD = 50.0
BEV_LATERAL = 25.0
BEV_SIZE = 480
CONF_THRESHOLD = 0.3


def create_depth_colormap(depth_uint16, scale, depth_h, depth_w, out_w, out_h):
    depth_m = depth_uint16.astype(np.float32) / 65535.0 * scale
    depth_norm = np.clip(depth_m / MAX_DEPTH_VIS * 255, 0, 255).astype(np.uint8)
    cmap = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_TURBO)
    return cv2.resize(cmap, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def create_bev(detections, focal_length, cx):
    bev = np.full((BEV_SIZE, BEV_SIZE, 3), (28, 24, 20), dtype=np.uint8)

    # Grid lines
    for dist in range(10, int(BEV_FORWARD) + 1, 10):
        y = int((1 - dist / BEV_FORWARD) * BEV_SIZE)
        cv2.line(bev, (0, y), (BEV_SIZE, y), (45, 40, 35), 1)
        cv2.putText(bev, f"{dist}m", (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)

    for offset_m in range(-20, 21, 10):
        x = int((offset_m / BEV_LATERAL * 0.5 + 0.5) * BEV_SIZE)
        cv2.line(bev, (x, 0), (x, BEV_SIZE), (45, 40, 35), 1)

    # Danger zone (0-10m) red tint
    danger_y = int((1 - 10.0 / BEV_FORWARD) * BEV_SIZE)
    red_tint = np.zeros_like(bev[danger_y:, :])
    red_tint[:, :, 2] = 35
    bev[danger_y:, :] = cv2.add(bev[danger_y:, :], red_tint)

    # Draw objects
    for det in detections:
        x1, y1, x2, y2, conf, cls_id, depth_m = det[0], det[1], det[2], det[3], det[4], int(det[5]), det[6]
        name = det[7] if len(det) > 7 else str(cls_id)

        if cls_id not in BEV_CLASSES or conf < CONF_THRESHOLD or depth_m <= 0.5:
            continue

        color = OBSTACLE_COLORS.get(cls_id, (name, (150, 150, 150)))[1]

        center_u = (x1 + x2) / 2
        forward = depth_m
        lateral = depth_m * (center_u - cx) / focal_length

        if forward > BEV_FORWARD or abs(lateral) > BEV_LATERAL:
            continue

        bev_x = int((lateral / BEV_LATERAL * 0.5 + 0.5) * BEV_SIZE)
        bev_y = int((1 - forward / BEV_FORWARD) * BEV_SIZE)

        obj_width_m = max(0.5, depth_m * (x2 - x1) / focal_length)
        obj_w_px = max(8, int(obj_width_m / (2 * BEV_LATERAL) * BEV_SIZE))
        obj_h_px = max(8, int(obj_w_px * 0.5))

        cv2.rectangle(bev,
                      (bev_x - obj_w_px // 2, bev_y - obj_h_px // 2),
                      (bev_x + obj_w_px // 2, bev_y + obj_h_px // 2),
                      color, -1)
        cv2.rectangle(bev,
                      (bev_x - obj_w_px // 2, bev_y - obj_h_px // 2),
                      (bev_x + obj_w_px // 2, bev_y + obj_h_px // 2),
                      (255, 255, 255), 1)

        label = f"{name} {depth_m:.0f}m"
        lx = bev_x + obj_w_px // 2 + 4
        ly = bev_y + 4
        if lx + len(label) * 6 > BEV_SIZE:
            lx = bev_x - obj_w_px // 2 - len(label) * 6 - 2
        cv2.putText(bev, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Ego triangle
    ex, ey = BEV_SIZE // 2, BEV_SIZE - 10
    pts = np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]])
    cv2.fillPoly(bev, [pts], (255, 255, 255))

    # Title
    cv2.putText(bev, "3D Object Map", (BEV_SIZE // 2 - 55, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Legend
    ly = 35
    for cls_id in sorted(BEV_CLASSES):
        if cls_id in OBSTACLE_COLORS:
            name, color = OBSTACLE_COLORS[cls_id]
            cv2.circle(bev, (BEV_SIZE - 70, ly), 4, color, -1, cv2.LINE_AA)
            cv2.putText(bev, name, (BEV_SIZE - 62, ly + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
            ly += 14

    return bev


def draw_yolo_overlay(frame_bgr, detections):
    overlay = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id, depth_m = det[0], det[1], det[2], det[3], det[4], int(det[5]), det[6]
        name = det[7] if len(det) > 7 else str(cls_id)

        if conf < CONF_THRESHOLD:
            continue

        color = OBSTACLE_COLORS.get(cls_id, (name, (180, 180, 180)))[1]
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # Semi-transparent fill for obstacles
        if cls_id in BEV_CLASSES:
            roi = overlay[iy1:iy2, ix1:ix2]
            tint = np.full_like(roi, color, dtype=np.uint8)
            overlay[iy1:iy2, ix1:ix2] = cv2.addWeighted(roi, 0.75, tint, 0.25, 0)

        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), color, 2)

        label = f"{name} {depth_m:.1f}m" if depth_m > 0 else name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(overlay, (ix1, iy1 - th - 6), (ix1 + tw + 4, iy1), color, -1)
        cv2.putText(overlay, label, (ix1 + 2, iy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def render_video(video_path, depth_npz_path, det_json_path, output_path, calib):
    print(f"\nRendering {os.path.basename(video_path)}...")

    data = np.load(depth_npz_path)
    depth_maps = data["depth_maps"]
    depth_scales = data["depth_scales"]
    num_depth = len(depth_maps)

    with open(det_json_path) as f:
        det_data = json.load(f)
    detections = det_data["frames"]
    vid_w = det_data["width"]
    vid_h = det_data["height"]
    depth_h, depth_w = depth_maps.shape[1], depth_maps.shape[2]

    focal = calib["intrinsics"]["focal_length"]
    cx = calib["intrinsics"]["cx"]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(total_frames, num_depth, len(detections))

    canvas_w = vid_w + vid_w + BEV_SIZE
    canvas_h = vid_h
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 30, (canvas_w, canvas_h))

    t_start = time.time()
    for i in range(num_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        dets = detections[i] if i < len(detections) else []

        # Panel 1: Original + YOLO overlay
        overlay = draw_yolo_overlay(frame_bgr, dets)

        # Panel 2: Depth colormap
        depth_color = create_depth_colormap(
            depth_maps[i], depth_scales[i], depth_h, depth_w, vid_w, vid_h
        )
        # Depth scale bar
        bar_x = vid_w - 30
        for sy in range(vid_h):
            d_val = int((1 - sy / vid_h) * 255)
            c = cv2.applyColorMap(np.array([[d_val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
            cv2.line(depth_color, (bar_x, sy), (bar_x + 15, sy), c.tolist(), 1)
        for d_m in range(0, int(MAX_DEPTH_VIS) + 1, 20):
            sy = int((1 - d_m / MAX_DEPTH_VIS) * (vid_h - 1))
            cv2.putText(depth_color, f"{d_m}m", (bar_x - 25, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

        # Panel 3: BEV
        bev = create_bev(dets, focal, cx)

        # Combine
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :vid_w] = overlay
        canvas[:, vid_w:vid_w * 2] = depth_color
        canvas[:, vid_w * 2:] = bev

        # Panel titles
        cv2.putText(canvas, "YOLOv11 Detections", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Depth (Metric)", (vid_w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Timestamp
        ts = i / 30.0
        cv2.putText(canvas, f"{ts:.1f}s", (vid_w - 55, vid_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Panel dividers
        cv2.line(canvas, (vid_w, 0), (vid_w, vid_h), (80, 80, 80), 1)
        cv2.line(canvas, (vid_w * 2, 0), (vid_w * 2, vid_h), (80, 80, 80), 1)

        writer.write(canvas)

        if (i + 1) % 300 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (num_frames - i - 1) / fps
            print(f"  [{i+1}/{num_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    writer.release()
    cap.release()

    render_time = time.time() - t_start
    print(f"  Rendered {num_frames} frames in {render_time:.1f}s ({num_frames/render_time:.1f} fps)")

    print(f"  Encoding H.264...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", output_path],
        capture_output=True,
    )
    os.remove(tmp_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved {output_path} ({size_mb:.1f} MB)")


def main():
    data_dir = str(DATA_DIR)
    depth_dir = str(DEPTH_DIR)
    calib = load_calib()

    for name in ["front-wide", "front-narrow"]:
        video_path = os.path.join(data_dir, f"{name}.mp4")
        depth_npz = os.path.join(depth_dir, name, "depth_maps.npz")
        det_json = os.path.join(depth_dir, name, "detections.json")
        output_path = os.path.join(depth_dir, f"{name}_3d.mp4")

        if not os.path.exists(depth_npz):
            print(f"Skipping {name} - no depth data (run cloud/depth_yolo.py first)")
            continue

        render_video(video_path, depth_npz, det_json, output_path, calib)

    print(f"\nDone! Videos saved to {depth_dir}/")


if __name__ == "__main__":
    main()
