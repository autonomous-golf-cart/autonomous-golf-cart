"""Render overlay, BEV, and side-by-side videos from saved segmentation maps.

Run after cloud/eval_segmentation.py. CPU only; re-run any time you change
visualization params without re-running inference.

    python tools/render_segmentation.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segdrive.bev import create_bev, create_overlay
from segdrive.paths import DATA_DIR, EVAL_DIR, load_calib
from segdrive.video import encode_h264, make_side_by_side


def main():
    video_path = str(DATA_DIR / "front-wide.mp4")
    output_dir = str(EVAL_DIR)
    calib = load_calib()

    cap = cv2.VideoCapture(video_path)

    for model_name in ["segformer", "mask2former", "ddrnet"]:
        model_dir = os.path.join(output_dir, model_name)
        seg_path = os.path.join(model_dir, "seg_maps.npz")
        if not os.path.exists(seg_path):
            print(f"Skipping {model_name} - no seg_maps.npz (run cloud/eval_segmentation.py first)")
            continue

        print(f"\nRendering {model_name}...")
        data = np.load(seg_path)
        seg_maps = data["seg_maps"]
        frame_indices = data["frame_indices"]

        overlay_frames = []
        bev_frames = []

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg_map = seg_maps[i]

            overlay = create_overlay(frame_rgb, seg_map)
            overlay_frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            bev = create_bev(seg_map, calib)
            bev_frames.append(cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))

            if (i + 1) % 50 == 0:
                print(f"  Rendered {i + 1}/{len(frame_indices)} frames")

        # Save videos
        overlay_path = os.path.join(model_dir, "overlay.mp4")
        bev_path = os.path.join(model_dir, "bev.mp4")
        sbs_path = os.path.join(model_dir, "side_by_side.mp4")

        print(f"  Encoding videos...")
        encode_h264(overlay_frames, overlay_path)
        encode_h264(bev_frames, bev_path)
        make_side_by_side(overlay_path, bev_path, sbs_path)

        # Save sample JPGs
        sample_indices = [0, len(overlay_frames)//4, len(overlay_frames)//2, 3*len(overlay_frames)//4]
        for j, si in enumerate(sample_indices):
            if si < len(overlay_frames):
                cv2.imwrite(os.path.join(model_dir, f"sample_{j}_overlay.jpg"),
                           overlay_frames[si], [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(os.path.join(model_dir, f"sample_{j}_bev.jpg"),
                           bev_frames[si], [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"  {model_name}: saved overlay.mp4, bev.mp4, side_by_side.mp4 + {len(sample_indices)} sample pairs")

    cap.release()
    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
