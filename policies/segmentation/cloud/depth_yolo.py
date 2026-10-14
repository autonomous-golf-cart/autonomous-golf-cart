"""Run Depth Anything V2 + YOLOv11 on Modal A100 for depth estimation and object detection."""
import modal
import sys
import os
import json
import zlib

app = modal.App("drive-by-depth")

depth_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.44.2",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "ultralytics",
    )
)


def _process_batch(batch_bgr, depth_processor, depth_model, yolo,
                   depth_h, depth_w, all_depth_maps, all_depth_scales, all_detections):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image

    batch_size = len(batch_bgr)
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in batch_bgr]

    # Depth estimation
    inputs = depth_processor(images=pil_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.float16)

    with torch.no_grad():
        depth_out = depth_model(pixel_values=pixel_values).predicted_depth

    depth_resized = F.interpolate(
        depth_out.unsqueeze(1).float(), size=(depth_h, depth_w),
        mode="bilinear", align_corners=False,
    ).squeeze(1).cpu().numpy()

    for i in range(batch_size):
        d = depth_resized[i]
        max_d = float(d.max()) if d.max() > 0 else 1.0
        all_depth_scales.append(max_d)
        all_depth_maps.append((d / max_d * 65535).clip(0, 65535).astype(np.uint16))

    # YOLO detection
    yolo_results = yolo(batch_bgr, verbose=False)

    for i, r in enumerate(yolo_results):
        frame_dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            bx1 = max(0, int(x1 / 2))
            by1 = max(0, int(y1 / 2))
            bx2 = min(depth_w, int(x2 / 2))
            by2 = min(depth_h, int(y2 / 2))
            if bx2 > bx1 and by2 > by1:
                median_depth = float(np.median(depth_resized[i, by1:by2, bx1:bx2]))
            else:
                median_depth = 0.0

            frame_dets.append([x1, y1, x2, y2, conf, cls_id, median_depth, r.names[cls_id]])
        all_detections.append(frame_dets)


@app.function(image=depth_image, gpu="A100", timeout=3600, memory=32768)
def run_depth_and_yolo(video_bytes: bytes, video_name: str):
    import torch
    import numpy as np
    import cv2
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from ultralytics import YOLO
    import time

    print("Loading Depth Anything V2 Metric Outdoor Large...")
    t0 = time.time()
    MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    depth_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).cuda().eval()
    params_m = sum(p.numel() for p in depth_model.parameters()) / 1e6
    print(f"  Loaded in {time.time()-t0:.1f}s ({params_m:.0f}M params)")

    print("Loading YOLOv11x...")
    t0 = time.time()
    yolo = YOLO("yolo11x.pt")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    depth_h, depth_w = height // 2, width // 2
    print(f"Video '{video_name}': {total_frames} frames, {video_fps}fps, {width}x{height}")
    print(f"Depth output: {depth_w}x{depth_h}")

    BATCH_SIZE = 12
    all_depth_maps = []
    all_depth_scales = []
    all_detections = []
    frame_count = 0
    t_start = time.time()
    batch_bgr = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_bgr.append(frame)

        if len(batch_bgr) == BATCH_SIZE:
            _process_batch(batch_bgr, depth_processor, depth_model, yolo,
                           depth_h, depth_w, all_depth_maps, all_depth_scales, all_detections)
            frame_count += len(batch_bgr)
            batch_bgr = []

            if frame_count % 240 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed
                eta = (total_frames - frame_count) / fps
                print(f"  [{frame_count}/{total_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    if batch_bgr:
        _process_batch(batch_bgr, depth_processor, depth_model, yolo,
                       depth_h, depth_w, all_depth_maps, all_depth_scales, all_detections)
        frame_count += len(batch_bgr)

    cap.release()
    elapsed = time.time() - t_start
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")

    depth_array = np.stack(all_depth_maps)
    depth_compressed = zlib.compress(depth_array.tobytes(), level=6)
    scales_array = np.array(all_depth_scales, dtype=np.float32)

    raw_mb = depth_array.nbytes / 1e6
    comp_mb = len(depth_compressed) / 1e6
    print(f"Depth: {raw_mb:.0f}MB raw -> {comp_mb:.0f}MB compressed ({raw_mb/comp_mb:.1f}x)")

    return {
        "depth_compressed": depth_compressed,
        "depth_shape": list(depth_array.shape),
        "depth_scales": scales_array.tobytes(),
        "num_scales": len(all_depth_scales),
        "detections": all_detections,
        "num_frames": frame_count,
        "video_fps": video_fps,
        "video_name": video_name,
        "width": width,
        "height": height,
        "elapsed_s": round(elapsed, 1),
    }


@app.local_entrypoint()
def main():
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segdrive.paths import DATA_DIR, DEPTH_DIR

    data_dir = str(DATA_DIR)
    output_dir = str(DEPTH_DIR)

    videos = {
        "front-wide": os.path.join(data_dir, "front-wide.mp4"),
        "front-narrow": os.path.join(data_dir, "front-narrow.mp4"),
    }

    handles = {}
    for name, path in videos.items():
        print(f"Reading {path}...")
        with open(path, "rb") as f:
            video_bytes = f.read()
        print(f"  {name}: {len(video_bytes)/1e6:.0f} MB")
        print(f"Launching {name} on A100...")
        handles[name] = run_depth_and_yolo.spawn(video_bytes, name)

    for name, handle in handles.items():
        print(f"\nWaiting for {name}...")
        result = handle.get()

        vid_dir = os.path.join(output_dir, name)
        os.makedirs(vid_dir, exist_ok=True)

        depth_array = np.frombuffer(
            zlib.decompress(result["depth_compressed"]), dtype=np.uint16
        ).reshape(result["depth_shape"])
        scales_array = np.frombuffer(result["depth_scales"], dtype=np.float32)

        np.savez_compressed(
            os.path.join(vid_dir, "depth_maps.npz"),
            depth_maps=depth_array,
            depth_scales=scales_array,
        )

        with open(os.path.join(vid_dir, "detections.json"), "w") as f:
            json.dump({
                "num_frames": result["num_frames"],
                "video_fps": result["video_fps"],
                "width": result["width"],
                "height": result["height"],
                "elapsed_s": result["elapsed_s"],
                "frames": result["detections"],
            }, f)

        depth_mb = depth_array.nbytes / 1e6
        print(f"  {name}: {result['num_frames']} frames, {result['elapsed_s']}s on A100")
        print(f"  Depth maps: {depth_mb:.0f}MB saved to {vid_dir}/depth_maps.npz")
        print(f"  Detections: {vid_dir}/detections.json")

    print(f"\nAll data saved to {output_dir}/")
    print("Run 'python3 tools/render_depth.py' to generate output videos.")
