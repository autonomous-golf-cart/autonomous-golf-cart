"""Render depth + YOLO and depth + seg videos on Modal CPU (much faster than local)."""
import modal
import sys
import os
import json

app = modal.App("drive-by-render-depth")

render_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy", "opencv-python-headless")
)


@app.function(image=render_image, cpu=8, memory=16384, timeout=1800)
def render_yolo_video(video_bytes: bytes, depth_npz_bytes: bytes, det_json_str: str,
                      calib_json_str: str, video_name: str):
    import numpy as np
    import cv2
    import subprocess
    import time
    import tempfile
    import io

    calib = json.loads(calib_json_str)
    detections_data = json.loads(det_json_str)
    detections = detections_data["frames"]

    depth_data = np.load(io.BytesIO(depth_npz_bytes))
    depth_maps = depth_data["depth_maps"]
    depth_scales = depth_data["depth_scales"]

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = min(total_frames, len(depth_maps), len(detections))

    focal = calib["intrinsics"]["focal_length"]
    cx = calib["intrinsics"]["cx"]

    OBSTACLE_COLORS = {
        0: ("person", (60, 60, 255)), 1: ("bicycle", (0, 165, 255)),
        2: ("car", (255, 150, 50)), 3: ("motorcycle", (0, 255, 255)),
        5: ("bus", (255, 100, 200)), 7: ("truck", (100, 220, 50)),
        9: ("traffic light", (30, 170, 250)), 11: ("stop sign", (50, 50, 255)),
        15: ("cat", (200, 150, 255)), 16: ("dog", (200, 150, 255)),
    }
    BEV_CLASSES = {0, 1, 2, 3, 5, 7, 15, 16}
    MAX_DEPTH = 80.0
    BEV_FWD = 50.0
    BEV_LAT = 25.0
    BEV_SZ = 480
    CONF = 0.3

    canvas_w = vid_w + vid_w + BEV_SZ
    tmp_out = "/tmp/output.tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, 30, (canvas_w, vid_h))

    # Precompute scale bar overlay (static across frames)
    bar_x = vid_w - 30
    scale_bar = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
    bar_vals = np.arange(vid_h, dtype=np.uint8)
    bar_vals = ((1 - np.arange(vid_h) / vid_h) * 255).clip(0, 255).astype(np.uint8)
    bar_strip = cv2.applyColorMap(bar_vals.reshape(-1, 1), cv2.COLORMAP_TURBO)
    bar_strip = np.repeat(bar_strip, 16, axis=1)
    scale_bar[:, bar_x:bar_x+16] = bar_strip[:, :min(16, vid_w - bar_x)]
    scale_bar_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
    scale_bar_mask[:, bar_x:bar_x+16] = 255
    for d_m in range(0, int(MAX_DEPTH) + 1, 20):
        sy = int((1 - d_m / MAX_DEPTH) * (vid_h - 1))
        cv2.putText(scale_bar, f"{d_m}m", (bar_x - 25, sy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        scale_bar_mask[max(0,sy-8):sy+8, bar_x-30:bar_x] = 255

    # Precompute BEV base (grid + danger zone + legend)
    bev_base = np.full((BEV_SZ, BEV_SZ, 3), (28, 24, 20), dtype=np.uint8)
    for dist in range(10, int(BEV_FWD) + 1, 10):
        y = int((1 - dist / BEV_FWD) * BEV_SZ)
        cv2.line(bev_base, (0, y), (BEV_SZ, y), (45, 40, 35), 1)
        cv2.putText(bev_base, f"{dist}m", (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
    for off in range(-20, 21, 10):
        x = int((off / BEV_LAT * 0.5 + 0.5) * BEV_SZ)
        cv2.line(bev_base, (x, 0), (x, BEV_SZ), (45, 40, 35), 1)
    dy = int((1 - 10.0 / BEV_FWD) * BEV_SZ)
    rt = np.zeros_like(bev_base[dy:, :]); rt[:, :, 2] = 35
    bev_base[dy:, :] = cv2.add(bev_base[dy:, :], rt)
    ex, ey = BEV_SZ // 2, BEV_SZ - 10
    pts = np.array([[ex, ey-14], [ex-7, ey], [ex+7, ey]])
    cv2.fillPoly(bev_base, [pts], (255, 255, 255))
    cv2.putText(bev_base, "3D Object Map", (BEV_SZ//2-55, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    ly = 35
    for cid in sorted(BEV_CLASSES):
        if cid in OBSTACLE_COLORS:
            n, clr = OBSTACLE_COLORS[cid]
            cv2.circle(bev_base, (BEV_SZ-70, ly), 4, clr, -1, cv2.LINE_AA)
            cv2.putText(bev_base, n, (BEV_SZ-62, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, clr, 1, cv2.LINE_AA)
            ly += 14

    t_start = time.time()
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections[i] if i < len(detections) else []

        # Overlay
        overlay = frame.copy()
        for det in dets:
            x1, y1, x2, y2, conf, cls_id, depth_m = det[0], det[1], det[2], det[3], det[4], int(det[5]), det[6]
            name = det[7] if len(det) > 7 else str(cls_id)
            if conf < CONF:
                continue
            color = OBSTACLE_COLORS.get(cls_id, (name, (180, 180, 180)))[1]
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
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

        # Depth colormap
        d_norm = np.clip(depth_maps[i].astype(np.float32) / 65535.0 * depth_scales[i] / MAX_DEPTH * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - d_norm, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (vid_w, vid_h))
        mask = scale_bar_mask > 0
        depth_color[mask] = scale_bar[mask]

        # BEV (copy precomputed base, draw dynamic objects)
        bev = bev_base.copy()

        for det in dets:
            x1, y1, x2, y2, conf, cls_id, depth_m = det[0], det[1], det[2], det[3], det[4], int(det[5]), det[6]
            name = det[7] if len(det) > 7 else str(cls_id)
            if cls_id not in BEV_CLASSES or conf < CONF or depth_m <= 0.5 or depth_m > BEV_FWD:
                continue
            color = OBSTACLE_COLORS.get(cls_id, (name, (150, 150, 150)))[1]
            cu = (x1 + x2) / 2
            lat = depth_m * (cu - cx) / focal
            if abs(lat) > BEV_LAT:
                continue
            bx = int((lat / BEV_LAT * 0.5 + 0.5) * BEV_SZ)
            by = int((1 - depth_m / BEV_FWD) * BEV_SZ)
            ow = max(0.5, depth_m * (x2 - x1) / focal)
            wp = max(8, int(ow / (2 * BEV_LAT) * BEV_SZ))
            hp = max(8, int(wp * 0.5))
            cv2.rectangle(bev, (bx - wp//2, by - hp//2), (bx + wp//2, by + hp//2), color, -1)
            cv2.rectangle(bev, (bx - wp//2, by - hp//2), (bx + wp//2, by + hp//2), (255, 255, 255), 1)
            lbl = f"{name} {depth_m:.0f}m"
            lx = bx + wp//2 + 4
            if lx + len(lbl) * 6 > BEV_SZ:
                lx = bx - wp//2 - len(lbl) * 6 - 2
            cv2.putText(bev, lbl, (lx, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        canvas = np.zeros((vid_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :vid_w] = overlay
        canvas[:, vid_w:vid_w*2] = depth_color
        canvas[:, vid_w*2:] = bev
        cv2.putText(canvas, "YOLOv11 Detections", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Depth (Metric)", (vid_w+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ts = i / 30.0
        cv2.putText(canvas, f"{ts:.1f}s", (vid_w-55, vid_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (vid_w, 0), (vid_w, vid_h), (80, 80, 80), 1)
        cv2.line(canvas, (vid_w*2, 0), (vid_w*2, vid_h), (80, 80, 80), 1)
        writer.write(canvas)

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (num_frames - i - 1) / fps
            print(f"  [{i+1}/{num_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    writer.release()
    cap.release()
    print(f"Rendered {num_frames} frames in {time.time()-t_start:.1f}s")

    out_path = "/tmp/output.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", tmp_out, "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "18", out_path], capture_output=True)
    with open(out_path, "rb") as f:
        return f.read()


@app.function(image=render_image, cpu=8, memory=16384, timeout=1800)
def render_seg_video(video_bytes: bytes, depth_npz_bytes: bytes, seg_npz_bytes: bytes,
                     calib_json_str: str, video_name: str):
    import numpy as np
    import cv2
    import subprocess
    import time
    import io

    calib = json.loads(calib_json_str)
    depth_data = np.load(io.BytesIO(depth_npz_bytes))
    depth_maps = depth_data["depth_maps"]
    depth_scales = depth_data["depth_scales"]
    seg_data = np.load(io.BytesIO(seg_npz_bytes))
    seg_maps = seg_data["seg_maps"]

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(total_frames, len(depth_maps), len(seg_maps))

    focal = calib["intrinsics"]["focal_length"]
    cx_param = calib["intrinsics"]["cx"]

    CITYSCAPES_COLORS = [
        (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
        (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
        (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
        (0,60,100),(0,80,100),(0,0,230),(119,11,32),
    ]
    OBSTACLE_SEG = {
        11: ("person",(60,60,255)), 12: ("rider",(0,100,255)),
        13: ("car",(255,150,50)), 14: ("truck",(100,220,50)),
        15: ("bus",(255,100,200)), 16: ("train",(200,150,50)),
        17: ("motorcycle",(0,255,255)), 18: ("bicycle",(0,165,255)),
    }
    MAX_DEPTH = 80.0
    BEV_FWD = 50.0
    BEV_LAT = 25.0
    BEV_SZ = 480
    MIN_AREA = 20

    canvas_w = vid_w + vid_w + BEV_SZ
    tmp_out = "/tmp/output.tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, 30, (canvas_w, vid_h))

    # Precompute scale bar overlay
    bar_x = vid_w - 30
    scale_bar = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
    bar_vals = ((1 - np.arange(vid_h) / vid_h) * 255).clip(0, 255).astype(np.uint8)
    bar_strip = cv2.applyColorMap(bar_vals.reshape(-1, 1), cv2.COLORMAP_TURBO)
    bar_strip = np.repeat(bar_strip, 16, axis=1)
    scale_bar[:, bar_x:bar_x+16] = bar_strip[:, :min(16, vid_w - bar_x)]
    scale_bar_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
    scale_bar_mask[:, bar_x:bar_x+16] = 255
    for dm in range(0, int(MAX_DEPTH) + 1, 20):
        sy = int((1 - dm / MAX_DEPTH) * (vid_h - 1))
        cv2.putText(scale_bar, f"{dm}m", (bar_x - 25, sy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        scale_bar_mask[max(0, sy-8):sy+8, bar_x-30:bar_x] = 255

    # Precompute BEV base (grid + danger zone + ego + legend)
    bev_base = np.full((BEV_SZ, BEV_SZ, 3), (28, 24, 20), dtype=np.uint8)
    for dist in range(10, int(BEV_FWD) + 1, 10):
        y = int((1 - dist / BEV_FWD) * BEV_SZ)
        cv2.line(bev_base, (0, y), (BEV_SZ, y), (45, 40, 35), 1)
        cv2.putText(bev_base, f"{dist}m", (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
    for off in range(-20, 21, 10):
        x = int((off / BEV_LAT * 0.5 + 0.5) * BEV_SZ)
        cv2.line(bev_base, (x, 0), (x, BEV_SZ), (45, 40, 35), 1)
    dy = int((1 - 10.0 / BEV_FWD) * BEV_SZ)
    rt = np.zeros_like(bev_base[dy:, :]); rt[:, :, 2] = 35
    bev_base[dy:, :] = cv2.add(bev_base[dy:, :], rt)
    ex, ey = BEV_SZ // 2, BEV_SZ - 10
    pts = np.array([[ex, ey - 14], [ex - 7, ey], [ex + 7, ey]])
    cv2.fillPoly(bev_base, [pts], (255, 255, 255))
    cv2.putText(bev_base, "3D Object Map (Seg)", (BEV_SZ // 2 - 70, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    ly = 35
    for cid in sorted(OBSTACLE_SEG.keys()):
        n, clr = OBSTACLE_SEG[cid]
        cv2.circle(bev_base, (BEV_SZ - 70, ly), 4, clr, -1, cv2.LINE_AA)
        cv2.putText(bev_base, n, (BEV_SZ - 62, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, clr, 1, cv2.LINE_AA)
        ly += 14

    # Precompute cityscapes color LUT for seg overlay
    cs_lut = np.zeros((19, 3), dtype=np.uint8)
    for cid, (r, g, b) in enumerate(CITYSCAPES_COLORS):
        cs_lut[cid] = (b, g, r)

    t_start = time.time()
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        seg = seg_maps[i]

        # Seg overlay (vectorized via LUT)
        seg_clipped = np.clip(seg, 0, 18)
        cmask = cs_lut[seg_clipped]
        overlay = cv2.addWeighted(frame, 0.55, cmask, 0.45, 0)

        # Depth colormap
        d_norm = np.clip(depth_maps[i].astype(np.float32) / 65535.0 * depth_scales[i] / MAX_DEPTH * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - d_norm, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (vid_w, vid_h))
        mask = scale_bar_mask > 0
        depth_color[mask] = scale_bar[mask]

        # BEV from seg + depth
        dh, dw = depth_maps[i].shape
        seg_half = cv2.resize(seg, (dw, dh), interpolation=cv2.INTER_NEAREST)
        obs_mask = np.isin(seg_half, list(OBSTACLE_SEG.keys())).astype(np.uint8)

        bev = bev_base.copy()

        if obs_mask.sum() > 0:
            d_full = depth_maps[i].astype(np.float32) / 65535.0 * depth_scales[i]
            nl, labels, stats, centroids = cv2.connectedComponentsWithStats(obs_mask, connectivity=8)
            for lid in range(1, nl):
                area = stats[lid, cv2.CC_STAT_AREA]
                if area < MIN_AREA:
                    continue
                comp = labels == lid
                cls_vals = seg_half[comp]
                dom = int(np.bincount(cls_vals, minlength=19).argmax())
                if dom not in OBSTACLE_SEG:
                    continue
                comp_d = d_full[comp]
                comp_d = comp_d[comp_d > 0.3]
                if len(comp_d) == 0:
                    continue
                med_d = float(np.median(comp_d))
                if med_d > BEV_FWD or med_d <= 0.5:
                    continue

                name, color = OBSTACLE_SEG[dom]
                cx_px = centroids[lid][0] * (vid_w / dw)
                lat = med_d * (cx_px - cx_param) / focal
                if abs(lat) > BEV_LAT:
                    continue
                bx = int((lat / BEV_LAT * 0.5 + 0.5) * BEV_SZ)
                by = int((1 - med_d / BEV_FWD) * BEV_SZ)
                ow_m = max(0.5, med_d * stats[lid, cv2.CC_STAT_WIDTH] * (vid_w/dw) / focal)
                wp = max(8, int(ow_m / (2*BEV_LAT) * BEV_SZ))
                hp = max(8, int(wp * 0.5))
                cv2.rectangle(bev, (bx-wp//2, by-hp//2), (bx+wp//2, by+hp//2), color, -1)
                cv2.rectangle(bev, (bx-wp//2, by-hp//2), (bx+wp//2, by+hp//2), (255,255,255), 1)
                lbl = f"{name} {med_d:.0f}m"
                lx = bx + wp//2 + 4
                if lx + len(lbl)*6 > BEV_SZ:
                    lx = bx - wp//2 - len(lbl)*6 - 2
                cv2.putText(bev, lbl, (lx, by+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        canvas = np.zeros((vid_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :vid_w] = overlay
        canvas[:, vid_w:vid_w*2] = depth_color
        canvas[:, vid_w*2:] = bev
        cv2.putText(canvas, "Mask2Former Segmentation", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Depth (Metric)", (vid_w+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        ts = i / 30.0
        cv2.putText(canvas, f"{ts:.1f}s", (vid_w-55, vid_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
        cv2.line(canvas, (vid_w, 0), (vid_w, vid_h), (80, 80, 80), 1)
        cv2.line(canvas, (vid_w*2, 0), (vid_w*2, vid_h), (80, 80, 80), 1)
        writer.write(canvas)

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (num_frames - i - 1) / fps
            print(f"  [{i+1}/{num_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    writer.release()
    cap.release()
    print(f"Rendered {num_frames} frames in {time.time()-t_start:.1f}s")

    out_path = "/tmp/output.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", tmp_out, "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "18", out_path], capture_output=True)
    with open(out_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segdrive.paths import CALIB_PATH, DATA_DIR, DEPTH_DIR

    data_dir = str(DATA_DIR)
    depth_dir = str(DEPTH_DIR)

    with open(CALIB_PATH) as f:
        calib_json = f.read()

    handles = {}

    for name in ["front-wide", "front-narrow"]:
        video_path = os.path.join(data_dir, f"{name}.mp4")
        depth_npz = os.path.join(depth_dir, name, "depth_maps.npz")
        det_json = os.path.join(depth_dir, name, "detections.json")
        seg_npz = os.path.join(depth_dir, name, "seg_maps_30fps.npz")

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        print(f"Read {name} video: {len(video_bytes)/1e6:.0f} MB")

        # YOLO render
        if os.path.exists(det_json):
            with open(depth_npz, "rb") as f:
                depth_bytes = f.read()
            with open(det_json) as f:
                det_str = f.read()
            print(f"  Launching YOLO render for {name} ({len(depth_bytes)/1e6:.0f}MB depth)...")
            handles[f"{name}_yolo"] = render_yolo_video.spawn(
                video_bytes, depth_bytes, det_str, calib_json, name
            )

        # Seg render (DAv2 depth)
        if os.path.exists(seg_npz) and os.path.exists(depth_npz):
            with open(depth_npz, "rb") as f:
                depth_bytes = f.read()
            with open(seg_npz, "rb") as f:
                seg_bytes = f.read()
            print(f"  Launching seg render for {name} ({len(seg_bytes)/1e6:.0f}MB seg)...")
            handles[f"{name}_seg"] = render_seg_video.spawn(
                video_bytes, depth_bytes, seg_bytes, calib_json, name
            )

        # DA3 seg render
        da3_depth = os.path.join(depth_dir, name, "da3_depth_maps.npz")
        da3_seg = os.path.join(depth_dir, name, "da3_seg_maps.npz")
        if os.path.exists(da3_depth) and os.path.exists(da3_seg):
            with open(da3_depth, "rb") as f:
                da3_depth_bytes = f.read()
            with open(da3_seg, "rb") as f:
                da3_seg_bytes = f.read()
            print(f"  Launching DA3+seg render for {name}...")
            handles[f"{name}_da3seg"] = render_seg_video.spawn(
                video_bytes, da3_depth_bytes, da3_seg_bytes, calib_json, name
            )

    for key, handle in handles.items():
        print(f"\nWaiting for {key}...")
        video_data = handle.get()
        if "_da3seg" in key:
            name = key.replace("_da3seg", "")
            out_name = f"{name}_da3_seg_3d.mp4"
        elif "_yolo" in key:
            name = key.replace("_yolo", "")
            out_name = f"{name.replace('-', '_')}_3d.mp4"
        else:
            name = key.replace("_seg", "")
            out_name = f"{name}_seg_3d.mp4"
        out_path = os.path.join(depth_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(video_data)
        print(f"  Saved {out_path} ({len(video_data)/1e6:.1f} MB)")

    print(f"\nAll videos saved to {depth_dir}/")
