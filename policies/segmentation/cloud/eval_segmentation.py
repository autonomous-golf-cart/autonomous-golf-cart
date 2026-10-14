"""Compare SegFormer, Mask2Former, and DDRNet on the front-wide drive video (Modal GPUs).

Writes per-model seg_maps.npz + metrics.json to outputs/eval/. Render them with
tools/render_segmentation.py.

    modal run cloud/eval_segmentation.py
"""
import modal
import sys
import json
import zlib
from pathlib import Path

app = modal.App("drive-by-segmentation-eval")

vol = modal.Volume.from_name("segmentation-data", create_if_missing=True)

segformer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "transformers==4.44.2",
        "Pillow",
        "numpy",
        "opencv-python-headless",
    )
)

mask2former_image = (
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

ddrnet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "Pillow",
        "numpy",
        "opencv-python-headless",
    )
    .run_commands(
        "git clone https://github.com/ydhongHIT/DDRNet.git /opt/DDRNet",
        "sed -i 's/BatchNorm2d = nn.SyncBatchNorm/BatchNorm2d = nn.BatchNorm2d/' /opt/DDRNet/segmentation/DDRNet_23_slim.py",
        "mkdir -p /opt/DDRNet/pretrained",
    )
    .add_local_file("ddrnet_23_slim_cityscapes.pth", "/opt/DDRNet/pretrained/ddrnet_23_slim.pth")
)

CITYSCAPES_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]


@app.function(image=segformer_image, gpu="A10G", timeout=1800, volumes={"/data": vol})
def run_segformer(video_bytes: bytes, sample_fps: int = 2):
    import torch
    import numpy as np
    import cv2
    import time
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from PIL import Image

    print("Loading SegFormer-B5...")
    t_load = time.time()
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    ).cuda().eval()
    load_time = time.time() - t_load
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded in {load_time:.1f}s. Params: {num_params:.1f}M")

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {video_fps} fps, sampling every {frame_interval}")

    seg_maps = []
    frame_indices = []
    frame_idx = 0
    infer_times = []
    class_pixel_counts = np.zeros(19, dtype=np.int64)

    torch.cuda.reset_peak_memory_stats()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_img, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        for c in range(19):
            class_pixel_counts[c] += (seg_map == c).sum()

        seg_maps.append(seg_map)
        frame_indices.append(frame_idx)

        if len(seg_maps) % 50 == 0:
            print(f"  Processed {len(seg_maps)} frames (frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
    avg_infer = np.mean(infer_times[1:]) if len(infer_times) > 1 else infer_times[0]
    total_pixels = class_pixel_counts.sum()
    class_pcts = {CITYSCAPES_LABELS[i]: round(float(class_pixel_counts[i] / total_pixels * 100), 2) for i in range(19) if class_pixel_counts[i] > 0}
    print(f"Done. {len(seg_maps)} frames. Avg: {avg_infer*1000:.1f}ms ({1/avg_infer:.1f} fps). Peak GPU: {peak_gpu:.0f}MB")

    seg_array = np.stack(seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)
    print(f"Seg maps: {seg_array.nbytes/1e6:.1f}MB -> {len(seg_compressed)/1e6:.1f}MB compressed")

    return {
        "seg_maps": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "frame_indices": frame_indices,
        "num_frames": len(seg_maps),
        "model": "segformer-b5",
        "params_m": round(num_params, 1),
        "avg_infer_ms": round(avg_infer * 1000, 1),
        "fps": round(1 / avg_infer, 1),
        "peak_gpu_mb": round(peak_gpu),
        "load_time_s": round(load_time, 1),
        "class_pcts": class_pcts,
    }


@app.function(image=mask2former_image, gpu="A10G", timeout=1800, volumes={"/data": vol})
def run_mask2former(video_bytes: bytes, sample_fps: int = 2):
    import torch
    import numpy as np
    import cv2
    import time
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    from PIL import Image

    print("Loading Mask2Former Swin-L...")
    t_load = time.time()
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    ).cuda().eval()
    load_time = time.time() - t_load
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded in {load_time:.1f}s. Params: {num_params:.1f}M")

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {video_fps} fps, sampling every {frame_interval}")

    seg_maps = []
    frame_indices = []
    frame_idx = 0
    infer_times = []
    class_pixel_counts = np.zeros(19, dtype=np.int64)

    torch.cuda.reset_peak_memory_stats()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_img, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        for c in range(19):
            class_pixel_counts[c] += (seg_map == c).sum()

        seg_maps.append(seg_map)
        frame_indices.append(frame_idx)

        if len(seg_maps) % 50 == 0:
            print(f"  Processed {len(seg_maps)} frames (frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
    avg_infer = np.mean(infer_times[1:]) if len(infer_times) > 1 else infer_times[0]
    total_pixels = class_pixel_counts.sum()
    class_pcts = {CITYSCAPES_LABELS[i]: round(float(class_pixel_counts[i] / total_pixels * 100), 2) for i in range(19) if class_pixel_counts[i] > 0}
    print(f"Done. {len(seg_maps)} frames. Avg: {avg_infer*1000:.1f}ms ({1/avg_infer:.1f} fps). Peak GPU: {peak_gpu:.0f}MB")

    seg_array = np.stack(seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)
    print(f"Seg maps: {seg_array.nbytes/1e6:.1f}MB -> {len(seg_compressed)/1e6:.1f}MB compressed")

    return {
        "seg_maps": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "frame_indices": frame_indices,
        "num_frames": len(seg_maps),
        "model": "mask2former-swin-l",
        "params_m": round(num_params, 1),
        "avg_infer_ms": round(avg_infer * 1000, 1),
        "fps": round(1 / avg_infer, 1),
        "peak_gpu_mb": round(peak_gpu),
        "load_time_s": round(load_time, 1),
        "class_pcts": class_pcts,
    }


@app.function(image=ddrnet_image, gpu="A10G", timeout=1800, volumes={"/data": vol})
def run_ddrnet(video_bytes: bytes, sample_fps: int = 2):
    import torch
    import numpy as np
    import cv2
    import time
    import sys
    sys.path.insert(0, "/opt/DDRNet")

    print("Loading DDRNet-23-slim...")
    t_load = time.time()
    from segmentation.DDRNet_23_slim import DualResNet, BasicBlock
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32,
                       spp_planes=128, head_planes=64, augment=False)
    ckpt = torch.load("/opt/DDRNet/pretrained/ddrnet_23_slim.pth", map_location="cpu")
    state = {k.replace("model.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    load_time = time.time() - t_load
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded in {load_time:.1f}s. Params: {num_params:.1f}M")

    video_path = "/tmp/input.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {video_fps} fps, sampling every {frame_interval}")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    seg_maps = []
    frame_indices = []
    frame_idx = 0
    infer_times = []
    class_pixel_counts = np.zeros(19, dtype=np.int64)

    torch.cuda.reset_peak_memory_stats()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]

        input_img = cv2.resize(frame_rgb, (1024, 512))
        input_tensor = torch.from_numpy(
            ((input_img / 255.0 - mean) / std).transpose(2, 0, 1)
        ).unsqueeze(0).float().cuda()

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if hasattr(output, 'logits'):
                output = output.logits
        torch.cuda.synchronize()
        infer_times.append(time.time() - t0)

        seg_map = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        seg_map = cv2.resize(seg_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        for c in range(19):
            class_pixel_counts[c] += (seg_map == c).sum()

        seg_maps.append(seg_map)
        frame_indices.append(frame_idx)

        if len(seg_maps) % 50 == 0:
            print(f"  Processed {len(seg_maps)} frames (frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6
    avg_infer = np.mean(infer_times[1:]) if len(infer_times) > 1 else infer_times[0]
    total_pixels = class_pixel_counts.sum()
    class_pcts = {CITYSCAPES_LABELS[i]: round(float(class_pixel_counts[i] / total_pixels * 100), 2) for i in range(19) if class_pixel_counts[i] > 0}
    print(f"Done. {len(seg_maps)} frames. Avg: {avg_infer*1000:.1f}ms ({1/avg_infer:.1f} fps). Peak GPU: {peak_gpu:.0f}MB")

    seg_array = np.stack(seg_maps)
    seg_compressed = zlib.compress(seg_array.tobytes(), level=6)
    print(f"Seg maps: {seg_array.nbytes/1e6:.1f}MB -> {len(seg_compressed)/1e6:.1f}MB compressed")

    return {
        "seg_maps": seg_compressed,
        "seg_shape": list(seg_array.shape),
        "frame_indices": frame_indices,
        "num_frames": len(seg_maps),
        "model": "ddrnet-23-slim",
        "params_m": round(num_params, 1),
        "avg_infer_ms": round(avg_infer * 1000, 1),
        "fps": round(1 / avg_infer, 1),
        "peak_gpu_mb": round(peak_gpu),
        "load_time_s": round(load_time, 1),
        "class_pcts": class_pcts,
    }


@app.local_entrypoint()
def main():
    import os
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from segdrive.paths import DATA_DIR, EVAL_DIR

    video_path = str(DATA_DIR / "front-wide.mp4")
    output_dir = str(EVAL_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading video from {video_path}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"Video size: {len(video_bytes) / 1e6:.1f} MB")

    models = {
        "segformer": run_segformer,
        "mask2former": run_mask2former,
        "ddrnet": run_ddrnet,
    }

    handles = {}
    for name, func in models.items():
        print(f"Launching {name}...")
        handles[name] = func.spawn(video_bytes, sample_fps=2)

    results = {}
    for name, handle in handles.items():
        print(f"\nWaiting for {name}...")
        result = handle.get()
        results[name] = result

        model_dir = os.path.join(output_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        seg_array = np.frombuffer(
            zlib.decompress(result["seg_maps"]), dtype=np.uint8
        ).reshape(result["seg_shape"])

        np.savez_compressed(
            os.path.join(model_dir, "seg_maps.npz"),
            seg_maps=seg_array,
            frame_indices=np.array(result["frame_indices"]),
        )

        # Save metrics
        metrics = {k: v for k, v in result.items() if k not in ("seg_maps", "seg_shape")}
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  {name}: {result['num_frames']} seg maps saved ({seg_array.nbytes/1e6:.1f}MB)")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON (A10G GPU, {list(results.values())[0]['num_frames']} frames @ 640x480)")
    print(f"{'='*80}")
    print(f"{'Metric':<22} {'SegFormer-B5':>16} {'Mask2Former-SwinL':>18} {'DDRNet-23-slim':>16}")
    print(f"{'-'*22} {'-'*16} {'-'*18} {'-'*16}")

    for key, label in [
        ("params_m", "Parameters (M)"),
        ("avg_infer_ms", "Avg Inference (ms)"),
        ("fps", "Throughput (fps)"),
        ("peak_gpu_mb", "Peak GPU (MB)"),
        ("load_time_s", "Load Time (s)"),
    ]:
        vals = []
        for name in ["segformer", "mask2former", "ddrnet"]:
            v = results[name].get(key, "N/A")
            vals.append(str(v))
        print(f"{label:<22} {vals[0]:>16} {vals[1]:>18} {vals[2]:>16}")

    print(f"\nAll seg maps saved to {output_dir}/")
    print(f"Run 'python3 tools/render_segmentation.py' to generate overlay, BEV, and side-by-side videos locally.")
