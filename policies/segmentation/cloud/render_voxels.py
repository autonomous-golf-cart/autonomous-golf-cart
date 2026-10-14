"""Render dense 3D voxel occupancy grid from depth + segmentation on Modal CPU.
Camera angle: slightly elevated behind car, looking forward — like the dashcam but zoomed out and tilted toward BEV.
Voxel volume: 30ft x 90ft x 10ft (9.1m x 27.4m x 3.0m).
Rendered at half resolution with nearest-neighbor upscale for blocky voxel look.
"""
import modal
import sys
import os
import json

app = modal.App("drive-by-voxel-render")

render_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy", "opencv-python-headless")
)


@app.function(image=render_image, cpu=8, memory=16384, timeout=3600)
def render_voxel_video(video_bytes: bytes, depth_npz_bytes: bytes, seg_npz_bytes: bytes,
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
    img_cx = calib["intrinsics"]["cx"]
    img_cy = calib["intrinsics"]["cy"]
    cam_h = calib["extrinsics"]["height_m"]
    pitch_rad = np.radians(calib["extrinsics"]["pitch_deg"])

    # Voxel grid: 30ft x 90ft x 10ft
    VOXEL_SIZE = 0.2
    X_MIN, X_MAX = -4.572, 4.572       # ±15ft lateral
    Y_MIN, Y_MAX = -3.048, 0.3         # 10ft above ground to 0.3m below (Y-neg = up)
    Z_MIN, Z_MAX = 1.5, 28.956         # ~5ft to ~95ft forward
    nx = int(np.ceil((X_MAX - X_MIN) / VOXEL_SIZE))
    ny = int(np.ceil((Y_MAX - Y_MIN) / VOXEL_SIZE))
    nz = int(np.ceil((Z_MAX - Z_MIN) / VOXEL_SIZE))
    half = VOXEL_SIZE / 2

    # Internal render at half res, upscale 2x for blocky voxel look
    IW, IH = 320, 240
    RENDER_W, RENDER_H = 640, 480
    canvas_w = vid_w + RENDER_W

    CLASS_COLORS = np.array([
        (128, 64, 128),   # 0  road
        (232, 35, 244),   # 1  sidewalk
        (70, 70, 70),     # 2  building
        (156, 102, 102),  # 3  wall
        (153, 153, 190),  # 4  fence
        (153, 153, 153),  # 5  pole
        (30, 170, 250),   # 6  traffic light
        (0, 220, 220),    # 7  traffic sign
        (35, 142, 107),   # 8  vegetation
        (152, 251, 152),  # 9  terrain
        (180, 130, 70),   # 10 sky
        (60, 20, 220),    # 11 person
        (0, 0, 255),      # 12 rider
        (142, 0, 0),      # 13 car
        (70, 0, 0),       # 14 truck
        (100, 60, 0),     # 15 bus
        (100, 80, 0),     # 16 train
        (230, 0, 0),      # 17 motorcycle
        (32, 11, 119),    # 18 bicycle
    ], dtype=np.uint8)

    skip_mask = np.zeros(19, dtype=bool)
    skip_mask[10] = True  # skip sky

    # Precompute pixel-to-ray table at depth map resolution
    dh, dw = depth_maps.shape[1], depth_maps.shape[2]
    sx_scale = vid_w / dw
    sy_scale = vid_h / dh
    u_coords = np.arange(dw) * sx_scale + sx_scale / 2
    v_coords = np.arange(dh) * sy_scale + sy_scale / 2
    uu, vv = np.meshgrid(u_coords, v_coords)
    ray_x = (uu - img_cx) / focal
    ray_y = (vv - img_cy) / focal
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    ray_x_rot = ray_x
    ray_y_rot = cos_p * ray_y + sin_p
    ray_z_rot = -sin_p * ray_y + cos_p

    # Render camera: behind and above car, looking forward and slightly down
    eye = np.array([0.0, -5.0, -3.0])
    target = np.array([0.0, -0.5, 14.0])
    up_world = np.array([0.0, -1.0, 0.0])

    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right)
    cam_down = np.cross(fwd, right)
    view_mat = np.array([right, cam_down, fwd])

    render_focal = 90.0
    icx, icy = IW / 2.0, IH / 2.0
    BG = np.array([15, 15, 15], dtype=np.uint8)

    # Precompute render base at output res (bounding box wireframe + legend)
    render_base = np.full((RENDER_H, RENDER_W, 3), BG, dtype=np.uint8)

    box_corners = np.array([
        [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MIN, Z_MIN],
        [X_MAX, Y_MAX, Z_MIN], [X_MIN, Y_MAX, Z_MIN],
        [X_MIN, Y_MIN, Z_MAX], [X_MAX, Y_MIN, Z_MAX],
        [X_MAX, Y_MAX, Z_MAX], [X_MIN, Y_MAX, Z_MAX],
    ])
    box_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    corners_cam = (box_corners - eye) @ view_mat.T
    out_focal = render_focal * 2
    out_cx, out_cy = RENDER_W / 2.0, RENDER_H / 2.0
    for i_e, j_e in box_edges:
        if corners_cam[i_e, 2] > 0.1 and corners_cam[j_e, 2] > 0.1:
            s1 = (int(corners_cam[i_e, 0] / corners_cam[i_e, 2] * out_focal + out_cx),
                  int(corners_cam[i_e, 1] / corners_cam[i_e, 2] * out_focal + out_cy))
            s2 = (int(corners_cam[j_e, 0] / corners_cam[j_e, 2] * out_focal + out_cx),
                  int(corners_cam[j_e, 1] / corners_cam[j_e, 2] * out_focal + out_cy))
            cv2.line(render_base, s1, s2, (40, 45, 40), 1, cv2.LINE_AA)

    cv2.putText(render_base, "3D Voxel Occupancy", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    legend = {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
              8: "vegetation", 9: "terrain", 11: "person", 13: "car", 14: "truck", 15: "bus"}
    ly = 40
    for cid in [0, 1, 2, 8, 9, 11, 13, 14, 15]:
        color = tuple(int(c) for c in CLASS_COLORS[cid])
        cv2.rectangle(render_base, (RENDER_W - 85, ly - 6), (RENDER_W - 75, ly + 4), color, -1)
        cv2.putText(render_base, legend[cid], (RENDER_W - 72, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
        ly += 14

    cs_lut = CLASS_COLORS.copy()

    tmp_out = "/tmp/voxel_output.tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, 30, (canvas_w, vid_h))

    t_start = time.time()
    for frame_i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        depth_m = depth_maps[frame_i].astype(np.float32) / 65535.0 * depth_scales[frame_i]
        seg = seg_maps[frame_i]
        seg_half = cv2.resize(seg, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # Back-project to 3D world coords
        pts_x = ray_x_rot * depth_m
        pts_y = ray_y_rot * depth_m - cam_h
        pts_z = ray_z_rot * depth_m

        # Filter
        seg_clipped = np.clip(seg_half, 0, 18)
        valid = (depth_m > 0.5) & (depth_m < Z_MAX + 2) & (~skip_mask[seg_clipped])
        px = pts_x[valid]
        py = pts_y[valid]
        pz = pts_z[valid]
        pc = seg_clipped[valid]

        # Voxelize
        vxi = ((px - X_MIN) / VOXEL_SIZE).astype(np.int32)
        vyi = ((py - Y_MIN) / VOXEL_SIZE).astype(np.int32)
        vzi = ((pz - Z_MIN) / VOXEL_SIZE).astype(np.int32)
        ib = (vxi >= 0) & (vxi < nx) & (vyi >= 0) & (vyi < ny) & (vzi >= 0) & (vzi < nz)
        vxi, vyi, vzi, pc = vxi[ib], vyi[ib], vzi[ib], pc[ib]

        voxel_grid = np.full((nx, ny, nz), -1, dtype=np.int8)
        voxel_grid[vxi, vyi, vzi] = pc

        occupied = np.argwhere(voxel_grid >= 0)
        color_buf = np.full((IH, IW, 3), BG, dtype=np.uint8)

        if len(occupied) > 0:
            occ_cls = voxel_grid[occupied[:, 0], occupied[:, 1], occupied[:, 2]]

            wx = occupied[:, 0].astype(np.float32) * VOXEL_SIZE + X_MIN + half
            wy = occupied[:, 1].astype(np.float32) * VOXEL_SIZE + Y_MIN + half
            wz = occupied[:, 2].astype(np.float32) * VOXEL_SIZE + Z_MIN + half

            pts_cam = (np.stack([wx, wy, wz], axis=1) - eye) @ view_mat.T
            in_front = pts_cam[:, 2] > 0.5
            pts_cam = pts_cam[in_front]
            occ_cls = occ_cls[in_front]
            wz_f = wz[in_front]

            if len(pts_cam) > 0:
                inv_z = 1.0 / pts_cam[:, 2]
                scr_x = (pts_cam[:, 0] * inv_z * render_focal + icx).astype(np.int32)
                scr_y = (pts_cam[:, 1] * inv_z * render_focal + icy).astype(np.int32)

                # Depth shading: near=bright, far=dim
                shade = np.clip(1.0 - (wz_f - Z_MIN) / (Z_MAX - Z_MIN) * 0.55, 0.45, 1.0)
                colors = (CLASS_COLORS[occ_cls].astype(np.float32) * shade[:, None]).clip(0, 255).astype(np.uint8)

                # Sort back-to-front (painter's algorithm)
                order = np.argsort(-pts_cam[:, 2])
                scr_x = scr_x[order]
                scr_y = scr_y[order]
                colors = colors[order]

                # Vectorized painting with 3x3 pixel expansion (center last for priority)
                for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),(0,0)]:
                    qx = scr_x + dx
                    qy = scr_y + dy
                    on = (qx >= 0) & (qx < IW) & (qy >= 0) & (qy < IH)
                    color_buf[qy[on], qx[on]] = colors[on]

        # Upscale 2x with nearest neighbor for blocky voxel aesthetic
        voxel_up = cv2.resize(color_buf, (RENDER_W, RENDER_H), interpolation=cv2.INTER_NEAREST)

        # Darken edges between voxels for 3D structure
        gray = cv2.cvtColor(voxel_up, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 15, 50)
        voxel_up[edges > 0] = (voxel_up[edges > 0].astype(np.int16) * 2 // 5).clip(0, 255).astype(np.uint8)

        # Composite onto render base
        render = render_base.copy()
        not_bg = ~np.all(voxel_up == BG, axis=2)
        render[not_bg] = voxel_up[not_bg]

        ts = frame_i / 30.0
        cv2.putText(render, f"{ts:.1f}s", (RENDER_W - 50, RENDER_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Seg overlay on original frame
        seg_c = np.clip(seg, 0, 18)
        overlay = cv2.addWeighted(frame, 0.55, cs_lut[seg_c], 0.45, 0)

        canvas = np.zeros((vid_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :vid_w] = overlay
        canvas[:, vid_w:] = render
        cv2.putText(canvas, "Segmentation", (10, vid_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (vid_w, 0), (vid_w, vid_h), (80, 80, 80), 1)
        writer.write(canvas)

        if (frame_i + 1) % 500 == 0 or frame_i == 0:
            elapsed = time.time() - t_start
            fps = (frame_i + 1) / elapsed
            eta = (num_frames - frame_i - 1) / fps
            print(f"  [{frame_i+1}/{num_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    writer.release()
    cap.release()
    print(f"Rendered {num_frames} frames in {time.time()-t_start:.1f}s")

    out_path = "/tmp/voxel_output.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", tmp_out, "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "18", out_path], capture_output=True)
    with open(out_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segdrive.paths import CALIB_PATH, DATA_DIR, DEPTH_DIR

    data_dir = str(DATA_DIR)
    depth_dir = str(DEPTH_DIR)

    with open(CALIB_PATH) as f:
        calib_json = f.read()

    handles = {}

    for name in ["front-wide", "front-narrow"]:
        video_path = os.path.join(data_dir, f"{name}.mp4")

        da3_depth = os.path.join(depth_dir, name, "da3_depth_maps.npz")
        da3_seg = os.path.join(depth_dir, name, "da3_seg_maps.npz")
        v2_depth = os.path.join(depth_dir, name, "depth_maps.npz")
        v2_seg = os.path.join(depth_dir, name, "seg_maps_30fps.npz")

        if os.path.exists(da3_depth) and os.path.exists(da3_seg):
            depth_path, seg_path = da3_depth, da3_seg
            suffix = "da3"
        elif os.path.exists(v2_depth) and os.path.exists(v2_seg):
            depth_path, seg_path = v2_depth, v2_seg
            suffix = "v2"
        else:
            print(f"Skipping {name} - no depth+seg data")
            continue

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        with open(depth_path, "rb") as f:
            depth_bytes = f.read()
        with open(seg_path, "rb") as f:
            seg_bytes = f.read()

        print(f"Launching voxel render for {name} ({suffix})...")
        handles[f"{name}_{suffix}"] = render_voxel_video.spawn(
            video_bytes, depth_bytes, seg_bytes, calib_json, name
        )

    for key, handle in handles.items():
        print(f"\nWaiting for {key}...")
        video_data = handle.get()
        out_path = os.path.join(depth_dir, f"{key.rsplit('_', 1)[0]}_voxel_{key.rsplit('_', 1)[1]}.mp4")
        with open(out_path, "wb") as f:
            f.write(video_data)
        print(f"  Saved {out_path} ({len(video_data)/1e6:.1f} MB)")

    print(f"\nAll voxel videos saved to {depth_dir}/")
