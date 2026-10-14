"""Where calibration, recorded drives, and generated outputs live.

Set ``SEGDRIVE_DATA_DIR`` to point at a recorded drive folder (front-wide.mp4,
front-narrow.mp4, gps.json, ego.jsonl, control.jsonl, map_route.json) stored
somewhere other than the default.
"""
import os
from pathlib import Path

SEG_ROOT = Path(__file__).resolve().parent.parent  # policies/segmentation/

CALIB_PATH = SEG_ROOT / "calibration" / "camera_calibration.json"

DATA_DIR = Path(os.environ.get(
    "SEGDRIVE_DATA_DIR",
    SEG_ROOT / "data" / "Caddy-Training-Data-2026-05-03_16-08-00",
))

OUTPUT_DIR = SEG_ROOT / "outputs"
EVAL_DIR = OUTPUT_DIR / "eval"              # per-model seg_maps.npz + renders
DEPTH_DIR = OUTPUT_DIR / "depth"            # per-camera depth / detections / seg
SEG_EXPORT_DIR = OUTPUT_DIR / "seg_export"  # RLE JSON + PNG export
MODELS_DIR = SEG_ROOT / "models"            # ONNX / TensorRT engines


def load_calib(path=CALIB_PATH):
    import json
    with open(path) as f:
        return json.load(f)
