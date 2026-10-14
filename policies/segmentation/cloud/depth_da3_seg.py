"""Run Depth Anything V3 + Mask2Former on Modal A100. Uses DA3-LARGE-1.1 for metric depth."""
import modal
import sys
import os
import json
import zlib

app = modal.App("drive-by-da3-seg")

da3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "xformers==0.0.27",
        "depth-anything-3",
        "transformers==4.44.2",
        "scipy",
        "Pillow",
        "numpy",
        "opencv-python-headless",
    )
)


@app.function(image=da3_image, gpu="A100", timeout=7200, memory=65536)
def run_da3_and_seg(video_bytes: bytes, video_name: str):
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    import time
    import tempfile

    # Load DA3
    print("Loading Depth Anything V3 (DA3-LARGE-1.1)...")
    t0 = time.time()
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
    da3 = da3.to(device="cuda")
    print(f"  DA3 loaded in {time.time()-t0:.1f}s")

    # Load Mask2Former
    print("Loading Mask2Former Swin-L...")
    t0 = time.time()
    seg_processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    ).cuda().eval()
    print(f"  Mask2Former loaded in {time.time()-t0:.1f}s")

    # Decode video
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

    all_depth_maps = []
    all_depth_scales = []
    all_seg_maps = []
    frame_count = 0
    t_start = time.time()

    DA3_BATCH = 8
    tmpdir = tempfile.mkdtemp()

    batch_bgr = []
    batch_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_bgr.append(frame)
        p = os.path.join(tmpdir, f"f{frame_count}.jpg")
        cv2.imwrite(p, frame)
        batch_paths.append(p)
        frame_count += 1

        if len(batch_bgr) == DA3_BATCH:
            _process_da3_seg_batch(
                batch_bgr, batch_paths, da3, seg_processor, seg_model,
                depth_h, depth_w, all_depth_maps, all_depth_scales, all_seg_maps
            )
            batch_bgr = []
            batch_paths = []

            if frame_count % 200 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed
                eta = (total_frames - frame_count) / fps
                print(f"  [{frame_count}/{total_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    if batch_bgr:
        _process_da3_seg_batch(
            batch_bgr, batch_paths, da3, seg_processor, seg_model,
            depth_h, depth_w, all_depth_maps, all_depth_scales, all_seg_maps
        )

    cap.release()
    elapsed = time.time() - t_start
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")

    # Pack results
    depth_array = np.stack(all_depth_maps)
    depth_compressed = zlib.compress(depth_array.tobytes(), level=6)
    scales_array = np.array(all_depth_scales, dtype=np.float32)

    seg_array = np.stack(all_seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)

    print(f"Depth: {depth_array.nbytes/1e6:.0f}MB -> {len(depth_compressed)/1e6:.0f}MB")
    print(f"Seg: {seg_array.nbytes/1e6:.0f}MB -> {len(seg_compressed)/1e6:.0f}MB")

    return {
        "depth_compressed": depth_compressed,
        "depth_shape": list(depth_array.shape),
        "depth_scales": scales_array.tobytes(),
        "seg_compressed": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "num_frames": frame_count,
        "video_fps": video_fps,
        "video_name": video_name,
        "width": width,
        "height": height,
        "elapsed_s": round(elapsed, 1),
    }


def _process_da3_seg_batch(batch_bgr, batch_paths, da3, seg_processor, seg_model,
                           depth_h, depth_w, all_depth_maps, all_depth_scales, all_seg_maps):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image

    batch_size = len(batch_bgr)

    # DA3 depth estimation
    prediction = da3.inference(batch_paths)
    depth_batch = prediction.depth  # [N, H, W] float32

    for i in range(batch_size):
        d = depth_batch[i].cpu().numpy() if hasattr(depth_batch[i], 'cpu') else depth_batch[i]
        if isinstance(d, np.ndarray) and d.ndim == 2:
            d_resized = cv2.resize(d.astype(np.float32), (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        else:
            d_resized = np.zeros((depth_h, depth_w), dtype=np.float32)
        max_d = float(d_resized.max()) if d_resized.max() > 0 else 1.0
        all_depth_scales.append(max_d)
        all_depth_maps.append((d_resized / max_d * 65535).clip(0, 65535).astype(np.uint16))

    # Mask2Former segmentation (one at a time for reliability)
    for i in range(batch_size):
        frame_rgb = cv2.cvtColor(batch_bgr[i], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        inputs = seg_processor(images=pil_img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = seg_model(**inputs)
        seg_map = seg_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)
        all_seg_maps.append(seg_map)


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
        handles[name] = run_da3_and_seg.spawn(video_bytes, name)

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
            os.path.join(vid_dir, "da3_depth_maps.npz"),
            depth_maps=depth_array,
            depth_scales=scales_array,
        )

        seg_array = np.frombuffer(
            zlib.decompress(result["seg_compressed"]), dtype=np.uint8
        ).reshape(result["seg_shape"])

        np.savez_compressed(
            os.path.join(vid_dir, "da3_seg_maps.npz"),
            seg_maps=seg_array,
        )

        print(f"  {name}: {result['num_frames']} frames in {result['elapsed_s']}s")
        print(f"  DA3 depth: {vid_dir}/da3_depth_maps.npz")
        print(f"  Seg maps: {vid_dir}/da3_seg_maps.npz")

    print(f"\nAll DA3 data saved to {output_dir}/")
    print("Run render on Modal: modal run cloud/render_depth.py --da3")
