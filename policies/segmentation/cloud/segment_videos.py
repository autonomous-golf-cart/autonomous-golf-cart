"""Run Mask2Former segmentation at 30fps on Modal A100. Pairs with depth data from cloud/depth_yolo.py."""
import modal
import sys
import os
import zlib

app = modal.App("drive-by-seg-30fps")

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


@app.function(image=seg_image, gpu="A100", timeout=3600, memory=32768)
def run_segmentation(video_bytes: bytes, video_name: str):
    import torch
    import numpy as np
    import cv2
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    from PIL import Image
    import time

    print("Loading Mask2Former Swin-L (Cityscapes)...")
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
    print(f"Video '{video_name}': {total_frames} frames, {video_fps}fps, {width}x{height}")

    seg_maps = []
    frame_count = 0
    t_start = time.time()

    torch.cuda.reset_peak_memory_stats()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        seg_maps.append(seg_map)
        frame_count += 1

        if frame_count % 200 == 0:
            elapsed = time.time() - t_start
            fps = frame_count / elapsed
            eta = (total_frames - frame_count) / fps
            print(f"  [{frame_count}/{total_frames}] {fps:.1f} fps, ETA {eta:.0f}s")

    cap.release()
    elapsed = time.time() - t_start
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"Peak GPU: {peak_gpu:.0f}MB")

    seg_array = np.stack(seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)
    raw_mb = seg_array.nbytes / 1e6
    comp_mb = len(seg_compressed) / 1e6
    print(f"Seg maps: {raw_mb:.0f}MB raw -> {comp_mb:.0f}MB compressed ({raw_mb/comp_mb:.1f}x)")

    return {
        "seg_compressed": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "num_frames": frame_count,
        "video_fps": video_fps,
        "video_name": video_name,
        "width": width,
        "height": height,
        "elapsed_s": round(elapsed, 1),
        "peak_gpu_mb": round(peak_gpu),
    }


@app.local_entrypoint()
def main():
    import numpy as np
    import json

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
        handles[name] = run_segmentation.spawn(video_bytes, name)

    for name, handle in handles.items():
        print(f"\nWaiting for {name}...")
        result = handle.get()

        vid_dir = os.path.join(output_dir, name)
        os.makedirs(vid_dir, exist_ok=True)

        seg_array = np.frombuffer(
            zlib.decompress(result["seg_compressed"]), dtype=np.uint8
        ).reshape(result["seg_shape"])

        np.savez_compressed(
            os.path.join(vid_dir, "seg_maps_30fps.npz"),
            seg_maps=seg_array,
        )

        seg_mb = seg_array.nbytes / 1e6
        print(f"  {name}: {result['num_frames']} frames in {result['elapsed_s']}s")
        print(f"  Seg maps: {seg_mb:.0f}MB saved to {vid_dir}/seg_maps_30fps.npz")
        print(f"  Peak GPU: {result['peak_gpu_mb']}MB")

    print(f"\nAll segmentation data saved to {output_dir}/")
    print("Run 'python3 tools/render_depth_seg.py' to generate output videos.")
