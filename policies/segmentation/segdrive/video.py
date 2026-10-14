"""H.264 encoding helpers (require ffmpeg on PATH)."""
import os
import subprocess

import cv2


def encode_h264(frames, path, fps=2):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", path],
        capture_output=True,
    )
    os.remove(tmp)


def make_side_by_side(overlay_path, bev_path, output_path):
    filter_str = (
        "[0:v]drawtext=text='Segmentation Overlay':"
        "fontsize=22:fontcolor=white:borderw=2:bordercolor=black:"
        "x=10:y=10[left];"
        "[1:v]scale=480:480[bev_scaled];"
        "[bev_scaled]drawtext=text='BEV (Top-Down)':"
        "fontsize=22:fontcolor=white:borderw=2:bordercolor=black:"
        "x=10:y=10[right];"
        "[left][right]hstack=inputs=2[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", overlay_path, "-i", bev_path,
         "-filter_complex", filter_str,
         "-map", "[out]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         output_path],
        capture_output=True,
    )
