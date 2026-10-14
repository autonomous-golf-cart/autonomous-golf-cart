"""Sanity-check the BEV projection with a synthetic segmentation map.

Paints a fake road / sidewalk / vegetation / sky layout over a frame, projects
it to BEV, and writes the side-by-side to outputs/debug/.

    python tools/debug_bev.py [frame.jpg]    # default: calibration/calibration_frame.jpg
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.bev import create_bev, create_overlay
from segdrive.paths import CALIB_PATH, OUTPUT_DIR, load_calib

calib = load_calib()
frame_path = sys.argv[1] if len(sys.argv) > 1 else str(CALIB_PATH.parent / "calibration_frame.jpg")
frame = cv2.imread(frame_path)
h, w = frame.shape[:2]

# Manual segmentation: road in lower portion, vegetation on sides, sky on top
seg = np.full((h, w), 2, dtype=np.uint8)  # building
seg[:int(h*0.35), :] = 10  # sky top
# Road: center-bottom area
for row in range(int(h*0.45), h):
    t = (row - h*0.45) / (h*0.55)
    road_left = int(w * (0.15 - t * 0.15))
    road_right = int(w * (0.85 + t * 0.15))
    seg[row, road_left:road_right] = 0  # road
# Vegetation strips on sides
seg[int(h*0.3):int(h*0.55), :int(w*0.35)] = 8
seg[int(h*0.3):int(h*0.55), int(w*0.65):] = 8
# Sidewalk edges
for row in range(int(h*0.5), h):
    t = (row - h*0.5) / (h*0.5)
    seg[row, max(0, int(w*(0.12-t*0.12))):int(w*(0.18-t*0.08))] = 1
    seg[row, int(w*(0.82+t*0.08)):min(w, int(w*(0.88+t*0.12)))] = 1

bev_color = create_bev(seg, calib)

# Create overlay
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
overlay = create_overlay(frame_rgb, seg, alpha=0.45)

# Side by side
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
bev_bgr = cv2.cvtColor(bev_color, cv2.COLOR_RGB2BGR)
bev_resized = cv2.resize(bev_bgr, (480, 480))
canvas = np.zeros((480, 640 + 480, 3), dtype=np.uint8)
canvas[:, :640] = overlay_bgr
canvas[:, 640:] = bev_resized
out_dir = OUTPUT_DIR / "debug"
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / "bev_side_by_side.jpg"), canvas)
cv2.imwrite(str(out_dir / "bev_only.jpg"), bev_bgr)
print(f"Saved {out_dir}/bev_side_by_side.jpg and bev_only.jpg")
