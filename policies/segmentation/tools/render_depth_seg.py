"""Client-side rendering of Mask2Former segmentation + depth into side-by-side videos with 3D BEV.
Requires cloud/depth_yolo.py and cloud/segment_videos.py to have been run first.
Re-run anytime you change visualization params without re-running inference.
"""
import numpy as np
import cv2
import os
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.cityscapes import CITYSCAPES_COLORS
from segdrive.paths import DATA_DIR, DEPTH_DIR, load_calib

# Cityscapes obstacle classes for self-driving (BGR colors for OpenCV)
OBSTACLE_SEG = {
    11: ("person",     (60, 20, 220)),
    12: ("rider",      (0, 0, 255)),
    13: ("car",        (142, 0, 0)),
    14: ("truck",      (70, 0, 0)),
    15: ("bus",        (100, 60, 0)),
    16: ("train",      (100, 80, 0)),
    17: ("motorcycle", (230, 0, 0)),
    18: ("bicycle",    (32, 11, 119)),
}

# Bright BEV colors (BGR)
BEV_COLORS = {
    11: ("person",     (60, 60, 255)),
    12: ("rider",      (0, 100, 255)),
    13: ("car",        (255, 150, 50)),
    14: ("truck",      (100, 220, 50)),
    15: ("bus",        (255, 100, 200)),
    16: ("train",      (200, 150, 50)),
    17: ("motorcycle", (0, 255, 255)),
    18: ("bicycle",    (0, 165, 255)),
}

MAX_DEPTH_VIS = 80.0
BEV_FORWARD = 50.0
BEV_LATERAL = 25.0
BEV_SIZE = 480
MIN_COMPONENT_AREA = 80


def create_seg_overlay(frame_bgr, seg_map, alpha=0.45):
    color_mask = np.zeros_like(frame_bgr)
    for cls_id, (r, g, b) in enumerate(CITYSCAPES_COLORS):
        mask = seg_map == cls_id
        color_mask[mask] = (b, g, r)
    return cv2.addWeighted(frame_bgr, 1 - alpha, color_mask, alpha, 0)


def create_depth_colormap(depth_uint16, scale, out_w, out_h):
    depth_m = depth_uint16.astype(np.float32) / 65535.0 * scale
    depth_norm = np.clip(depth_m / MAX_DEPTH_VIS * 255, 0, 255).astype(np.uint8)
    cmap = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_TURBO)
    return cv2.resize(cmap, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def extract_obstacles(seg_map, depth_m_half):
    """Find obstacle objects from segmentation, return list of (cls_id, name, cx_pixel, area, median_depth)."""
    obstacle_ids = set(OBSTACLE_SEG.keys())
    h, w = seg_map.shape

    # Downsample seg to half resolution to match depth
    dh, dw = depth_m_half.shape
    seg_half = cv2.resize(seg_map, (dw, dh), interpolation=cv2.INTER_NEAREST)

    obstacle_mask = np.isin(seg_half, list(obstacle_ids)).astype(np.uint8)
    if obstacle_mask.sum() == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        obstacle_mask, connectivity=8
    )

    objects = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA // 4:
            continue

        component_mask = labels == label_id
        cls_values = seg_half[component_mask]
        counts = np.bincount(cls_values, minlength=19)
        dominant_cls = int(np.argmax(counts))

        if dominant_cls not in obstacle_ids:
            continue

        depth_vals = depth_m_half[component_mask]
        depth_vals = depth_vals[depth_vals > 0.3]
        if len(depth_vals) == 0:
            continue
        median_depth = float(np.median(depth_vals))

        cx_half = centroids[label_id][0]
        cx_full = cx_half * (w / dw)

        x = stats[label_id, cv2.CC_STAT_LEFT]
        obj_w = stats[label_id, cv2.CC_STAT_WIDTH]

        name = OBSTACLE_SEG[dominant_cls][0]
        objects.append((dominant_cls, name, cx_full, obj_w * (w / dw), area * 4, median_depth))

    return objects


def create_bev(obstacles, focal_length, cx):
    bev = np.full((BEV_SIZE, BEV_SIZE, 3), (28, 24, 20), dtype=np.uint8)

    for dist in range(10, int(BEV_FORWARD) + 1, 10):
        y = int((1 - dist / BEV_FORWARD) * BEV_SIZE)
        cv2.line(bev, (0, y), (BEV_SIZE, y), (45, 40, 35), 1)
        cv2.putText(bev, f"{dist}m", (5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)

    for offset_m in range(-20, 21, 10):
        x = int((offset_m / BEV_LATERAL * 0.5 + 0.5) * BEV_SIZE)
        cv2.line(bev, (x, 0), (x, BEV_SIZE), (45, 40, 35), 1)

    # Danger zone
    danger_y = int((1 - 10.0 / BEV_FORWARD) * BEV_SIZE)
    red_tint = np.zeros_like(bev[danger_y:, :])
    red_tint[:, :, 2] = 35
    bev[danger_y:, :] = cv2.add(bev[danger_y:, :], red_tint)

    for cls_id, name, cx_pixel, obj_w_px, area, depth_m in obstacles:
        if depth_m <= 0.5 or depth_m > BEV_FORWARD:
            continue

        color = BEV_COLORS.get(cls_id, (name, (150, 150, 150)))[1]

        lateral = depth_m * (cx_pixel - cx) / focal_length
        if abs(lateral) > BEV_LATERAL:
            continue

        bev_x = int((lateral / BEV_LATERAL * 0.5 + 0.5) * BEV_SIZE)
        bev_y = int((1 - depth_m / BEV_FORWARD) * BEV_SIZE)

        obj_width_m = max(0.5, depth_m * obj_w_px / focal_length)
        w_px = max(8, int(obj_width_m / (2 * BEV_LATERAL) * BEV_SIZE))
        h_px = max(8, int(w_px * 0.5))

        cv2.rectangle(bev, (bev_x - w_px // 2, bev_y - h_px // 2),
                      (bev_x + w_px // 2, bev_y + h_px // 2), color, -1)
        cv2.rectangle(bev, (bev_x - w_px // 2, bev_y - h_px // 2),
                      (bev_x + w_px // 2, bev_y + h_px // 2), (255, 255, 255), 1)

        label = f"{name} {depth_m:.0f}m"
        lx = bev_x + w_px // 2 + 4
        if lx + len(label) * 6 > BEV_SIZE:
            lx = bev_x - w_px // 2 - len(label) * 6 - 2
        cv2.putText(bev, label, (lx, bev_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Ego
    ex, ey = BEV_SIZE // 2, BEV_SIZE - 10
    pts = np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]])
    cv2.fillPoly(bev, [pts], (255, 255, 255))

    cv2.putText(bev, "3D Object Map (Seg)", (BEV_SIZE // 2 - 70, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    ly = 35
    for cls_id in sorted(BEV_COLORS.keys()):
        name, color = BEV_COLORS[cls_id]
        cv2.circle(bev, (BEV_SIZE - 70, ly), 4, color, -1, cv2.LINE_AA)
        cv2.putText(bev, name, (BEV_SIZE - 62, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
        ly += 14

    return bev


def render_video(video_path, depth_npz_path, seg_npz_path, output_path, calib):
    print(f"\nRendering {os.path.basename(video_path)} (seg+depth)...")

    depth_data = np.load(depth_npz_path)
    depth_maps = depth_data["depth_maps"]
    depth_scales = depth_data["depth_scales"]

    seg_data = np.load(seg_npz_path)
    seg_maps = seg_data["seg_maps"]

    depth_h, depth_w = depth_maps.shape[1], depth_maps.shape[2]
    focal = calib["intrinsics"]["focal_length"]
    cx = calib["intrinsics"]["cx"]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = min(total_frames, len(depth_maps), len(seg_maps))

    canvas_w = vid_w + vid_w + BEV_SIZE
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, 30, (canvas_w, vid_h))

    t_start = time.time()
    for i in range(num_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        seg_map = seg_maps[i]

        # Panel 1: Segmentation overlay
        overlay = create_seg_overlay(frame_bgr, seg_map)

        # Panel 2: Depth colormap
        depth_color = create_depth_colormap(depth_maps[i], depth_scales[i], vid_w, vid_h)

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

        # Panel 3: BEV from segmentation + depth
        depth_m_half = depth_maps[i].astype(np.float32) / 65535.0 * depth_scales[i]
        obstacles = extract_obstacles(seg_map, depth_m_half)
        bev = create_bev(obstacles, focal, cx)

        # Combine
        canvas = np.zeros((vid_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :vid_w] = overlay
        canvas[:, vid_w:vid_w * 2] = depth_color
        canvas[:, vid_w * 2:] = bev

        cv2.putText(canvas, "Mask2Former Segmentation", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Depth (Metric)", (vid_w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ts = i / 30.0
        cv2.putText(canvas, f"{ts:.1f}s", (vid_w - 55, vid_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

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

    print("  Encoding H.264...")
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
        seg_npz = os.path.join(depth_dir, name, "seg_maps_30fps.npz")
        output_path = os.path.join(depth_dir, f"{name}_seg_3d.mp4")

        if not os.path.exists(depth_npz):
            print(f"Skipping {name} - no depth data (run cloud/depth_yolo.py first)")
            continue
        if not os.path.exists(seg_npz):
            print(f"Skipping {name} - no seg data (run cloud/segment_videos.py first)")
            continue

        render_video(video_path, depth_npz, seg_npz, output_path, calib)

    print(f"\nDone! Videos saved to {depth_dir}/")


if __name__ == "__main__":
    main()
