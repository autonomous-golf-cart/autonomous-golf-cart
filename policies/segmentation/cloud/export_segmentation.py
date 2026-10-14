"""Run Mask2Former on H100 for 30s of video, export segmentation as images + JSON (RLE-encoded).
Output: per-frame PNGs, overlay video, and a JSON file with RLE-encoded masks for reconstruction.
"""
import modal
import sys
import os
import zlib
import json

app = modal.App("drive-by-seg-export")

seg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.44.2",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "scipy",
    )
)

CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]

CITYSCAPES_COLORS = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
]


def rle_encode(mask):
    """Run-length encode a binary or class-ID mask (flattened row-major).
    Returns list of [value, count] pairs."""
    flat = mask.flatten()
    runs = []
    if len(flat) == 0:
        return runs
    current_val = int(flat[0])
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == current_val:
            count += 1
        else:
            runs.append([current_val, count])
            current_val = int(flat[i])
            count = 1
    runs.append([current_val, count])
    return runs


@app.function(image=seg_image, gpu="H100", timeout=1800, memory=32768)
def run_segmentation_export(video_bytes: bytes, duration_s: float = 30.0):
    import torch
    import numpy as np
    import cv2
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    from PIL import Image
    import time

    print("Loading Mask2Former Swin-L (Cityscapes) on H100...")
    t0 = time.time()
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    ).cuda().eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {time.time()-t0:.1f}s ({params_m:.0f}M params)")

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(duration_s * video_fps)
    print(f"Video: {total_frames} frames, {video_fps}fps, {width}x{height}")
    print(f"Processing first {duration_s}s = {max_frames} frames")

    torch.cuda.reset_peak_memory_stats()

    seg_maps = []
    frames_rgb = []
    frame_count = 0
    t_start = time.time()
    infer_times = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_img, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t_inf = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        infer_times.append(time.time() - t_inf)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        seg_maps.append(seg_map)
        frames_rgb.append(frame_rgb)
        frame_count += 1

        if frame_count % 100 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed
            print(f"  [{frame_count}/{max_frames}] {fps:.1f} fps")

    cap.release()
    elapsed = time.time() - t_start
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
    avg_infer = np.mean(infer_times[1:]) * 1000 if len(infer_times) > 1 else infer_times[0] * 1000
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"Avg inference: {avg_infer:.1f}ms, Peak GPU: {peak_gpu:.0f}MB")

    # Build RLE-encoded JSON data
    print("Encoding segmentation as RLE JSON...")
    rle_frames = []
    class_pixel_counts = np.zeros(19, dtype=np.int64)
    for i, seg_map in enumerate(seg_maps):
        rle = rle_encode(seg_map)
        rle_frames.append(rle)
        for c in range(19):
            class_pixel_counts[c] += (seg_map == c).sum()

    total_pixels = class_pixel_counts.sum()
    class_pcts = {}
    for i in range(19):
        if class_pixel_counts[i] > 0:
            class_pcts[CITYSCAPES_LABELS[i]] = round(float(class_pixel_counts[i] / total_pixels * 100), 2)

    seg_json = {
        "metadata": {
            "model": "mask2former-swin-large-cityscapes-semantic",
            "params_m": round(params_m, 1),
            "width": width,
            "height": height,
            "num_frames": frame_count,
            "video_fps": video_fps,
            "duration_s": round(duration_s, 1),
            "avg_infer_ms": round(avg_infer, 1),
            "peak_gpu_mb": round(peak_gpu),
            "encoding": "rle",
            "rle_description": "Each frame is a list of [class_id, pixel_count] pairs in row-major order. To reconstruct: create array of shape (height, width), fill sequentially with class_id repeated pixel_count times.",
            "class_labels": CITYSCAPES_LABELS,
            "class_colors_rgb": CITYSCAPES_COLORS,
        },
        "class_distribution": class_pcts,
        "frames": rle_frames,
    }

    seg_json_bytes = json.dumps(seg_json).encode()
    print(f"JSON size: {len(seg_json_bytes)/1e6:.1f}MB")

    # Build per-frame PNG masks (class ID as grayscale)
    print("Encoding per-frame PNGs...")
    png_frames = []
    for seg_map in seg_maps:
        _, png_buf = cv2.imencode(".png", seg_map)
        png_frames.append(png_buf.tobytes())

    # Build overlay video
    print("Encoding overlay video...")
    overlay_path = "/tmp/overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, video_fps, (width, height))
    for frame_rgb, seg_map in zip(frames_rgb, seg_maps):
        color_mask = np.zeros_like(frame_rgb)
        for cls_id, color in enumerate(CITYSCAPES_COLORS):
            mask = seg_map == cls_id
            color_mask[mask] = color
        overlay = ((0.55 * frame_rgb) + (0.45 * color_mask)).astype(np.uint8)
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    writer.release()

    with open(overlay_path, "rb") as f:
        overlay_bytes = f.read()
    print(f"Overlay video: {len(overlay_bytes)/1e6:.1f}MB")

    # Compress seg maps for npz
    seg_array = np.stack(seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)

    return {
        "seg_json": seg_json_bytes,
        "png_frames": png_frames,
        "overlay_video": overlay_bytes,
        "seg_compressed": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "num_frames": frame_count,
        "width": width,
        "height": height,
        "video_fps": video_fps,
    }


@app.local_entrypoint()
def main():
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segdrive.paths import DATA_DIR, SEG_EXPORT_DIR

    data_dir = str(DATA_DIR)
    video_path = os.path.join(data_dir, "front-wide.mp4")
    output_dir = str(SEG_EXPORT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading {video_path}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"  Size: {len(video_bytes)/1e6:.0f} MB")

    print("Launching segmentation on H100 (30s clip)...")
    result = run_segmentation_export.remote(video_bytes, duration_s=30.0)

    # Save JSON with RLE-encoded segmentation
    json_path = os.path.join(output_dir, "segmentation.json")
    with open(json_path, "wb") as f:
        f.write(result["seg_json"])
    print(f"Saved RLE JSON: {json_path} ({len(result['seg_json'])/1e6:.1f}MB)")

    # Save per-frame PNGs
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, png_bytes in enumerate(result["png_frames"]):
        with open(os.path.join(frames_dir, f"{i:05d}.png"), "wb") as f:
            f.write(png_bytes)
    print(f"Saved {len(result['png_frames'])} frame PNGs to {frames_dir}/")

    # Save overlay video
    overlay_path = os.path.join(output_dir, "overlay.mp4")
    with open(overlay_path, "wb") as f:
        f.write(result["overlay_video"])
    print(f"Saved overlay video: {overlay_path} ({len(result['overlay_video'])/1e6:.1f}MB)")

    # Save npz for numpy workflows
    seg_array = np.frombuffer(
        zlib.decompress(result["seg_compressed"]), dtype=np.uint8
    ).reshape(result["seg_shape"])
    npz_path = os.path.join(output_dir, "seg_maps.npz")
    np.savez_compressed(npz_path, seg_maps=seg_array)
    print(f"Saved npz: {npz_path} ({os.path.getsize(npz_path)/1e6:.1f}MB)")

    print(f"\n{'='*60}")
    print(f"All outputs saved to {output_dir}/")
    print(f"  segmentation.json  - RLE-encoded per-frame masks (reconstructable)")
    print(f"  frames/            - Per-frame PNG class-ID masks")
    print(f"  overlay.mp4        - Visual overlay video")
    print(f"  seg_maps.npz       - Numpy array for programmatic use")
    print(f"{'='*60}")
