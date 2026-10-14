"""SegFormer (Cityscapes) loading and single-frame inference via transformers."""
import time

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_VARIANTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_segformer(variant, device):
    name = MODEL_VARIANTS[variant]
    print(f"Loading {name} on {device}...")
    t0 = time.time()
    proc = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {time.time()-t0:.1f}s ({params:.1f}M params)")
    return proc, model


def segment_frame(frame_rgb, proc, model, device):
    pil = Image.fromarray(frame_rgb)
    inputs = proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    seg = proc.post_process_semantic_segmentation(
        out, target_sizes=[frame_rgb.shape[:2]]
    )[0].cpu().numpy().astype(np.uint8)
    return seg

