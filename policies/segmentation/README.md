# Drive by Segmentation

Camera-only driving. A single wide-angle front camera is segmented with
**SegFormer** (Cityscapes), projected to a **bird's-eye view** with the
calibrated fisheye model, and a **lane-aware centerline planner** turns the
drivable area into a trajectory and a steering command.

<p align="center">
  <img src="../../docs/assets/segmentation-demo.gif" width="560" alt="SegFormer road segmentation on Stanford campus">
</p>

## Pipeline

```
camera frame ─▶ SegFormer (b0/b2/b5) ─▶ class map ─▶ BEV projection ─▶ drivable mask
                                                                         │
            GPS route (optional) ─▶ map_route_to_bev ─────────────────────┤
                                                                         ▼
                              steering ◀─ steering estimator ◀─ lane-aware centerline path
```

Each box maps to one module in [`segdrive/`](segdrive):
`segformer` → `bev` → `planning` → `steering`, with `draw` for visualization.

## Layout

```
segmentation/
├── live.py            Run the full pipeline on a video or camera, with a live display
├── segdrive/          Shared library — everything else imports from here
├── calibration/       Camera calibration + the browser tool for tuning it
├── onboard/           On-cart runtime for the Jetson (planner sidecar, TensorRT)
├── tools/             Local CPU scripts: render videos, debug the planner, 3D viewer
├── cloud/             GPU batch jobs on Modal (segmentation, depth, rendering)
├── data/              (git-ignored) recorded drives
├── outputs/           (git-ignored) everything the tools and cloud jobs generate
└── models/            (git-ignored) ONNX / TensorRT engines
```

### `segdrive/`: the library

| Module | What it does |
|--------|--------------|
| `segformer.py` | Load SegFormer b0/b2/b5 and segment one frame. |
| `cityscapes.py` | The 19 Cityscapes class labels and colors. |
| `bev.py` | `create_bev` fisheye → bird's-eye-view projection, `create_overlay`, `road_mask_from_bev`. |
| `planning.py` | Lane-aware centerline planner (used for driving), plus Frenet and MPC (bicycle model) planners and GPS/map-route helpers. |
| `steering.py` | `SteeringEstimator`: BEV path heading → steering-column angle. Also reads recorded control logs for ground truth. |
| `draw.py` | Draws trajectories, routes, lookahead, and the steering wheel on BEV frames. `set_bev_range(calib)` sets the BEV scale. |
| `video.py` | H.264 encoding through ffmpeg. |
| `paths.py` | Where calibration, data, and outputs live. |

### `calibration/`

| File | What it does |
|------|--------------|
| `camera_calibration.json` | Fisheye intrinsics, mounting extrinsics (ELP 170° camera, 1.78 m high), BEV range, road width. |
| `bev_tuner.html` | Open in a browser to tune the BEV projection by eye, then export a new `camera_calibration.json`. |
| `calibration_frame.jpg` | Reference frame from the front camera for the tuner. |

### `onboard/`: on the cart (Jetson AGX Thor)

| File | What it does |
|------|--------------|
| `segmentation_infer.py` | Live planner sidecar: reads the cameras, publishes `steer_deg` + gas/brake targets. |
| `drive.py` | Reads the sidecar's targets at 50 Hz and drives the cart through `cartlib`. Dry run unless `--i-understand-this-drives`. |
| `seg_fast.py` | Precomputed BEV remap and a cv2-accelerated planner, about 10× faster than `segdrive`. |
| `export_segformer_trt.py` | Export SegFormer to ONNX and build a TensorRT engine into `models/`. |
| `seg_depth_pointcloud.py` | Segmented RealSense/depth point cloud with a live viewer. |

### `tools/`: local CPU scripts

| Script | What it does | Needs |
|--------|--------------|-------|
| `render_segmentation.py` | Overlay, BEV, and side-by-side videos for each model. | `cloud/eval_segmentation.py` |
| `render_trajectories.py` | Planned trajectory + predicted vs. ground-truth steering video. | `cloud/eval_segmentation.py` |
| `render_depth.py` | Depth + YOLO detections with a 3D object map. | `cloud/depth_yolo.py` |
| `render_depth_seg.py` | Depth + segmentation with a 3D obstacle map. | `cloud/depth_yolo.py`, `cloud/segment_videos.py` |
| `voxel_viewer.py` | Interactive in-browser 3D point cloud colored by class. | depth + seg outputs |
| `debug_bev.py` | Sanity check of the BEV projection on a synthetic segmentation. | nothing |
| `debug_lane_planner.py` | Planner debug images on a few frames. | `cloud/eval_segmentation.py` |

### `cloud/`: GPU jobs on [Modal](https://modal.com)

| Job | What it does | Output |
|-----|--------------|--------|
| `eval_segmentation.py` | SegFormer vs. Mask2Former vs. DDRNet on the drive video. | `outputs/eval/` |
| `segment_videos.py` | Mask2Former at 30 fps on both front cameras. | `outputs/depth/` |
| `export_segmentation.py` | Mask2Former export as PNG + RLE JSON. | `outputs/seg_export/` |
| `depth_yolo.py` | Depth Anything V2 + YOLOv11. | `outputs/depth/` |
| `depth_da3_seg.py` | Depth Anything V3 + Mask2Former. | `outputs/depth/` |
| `render_trajectories.py` | `tools/render_trajectories.py`, run on a cloud CPU. | `outputs/eval/trajectories/` |
| `render_depth.py` | `tools/render_depth*.py`, run on a cloud CPU. | `outputs/depth/` |
| `render_voxels.py` | 3D voxel occupancy video. | `outputs/depth/` |

## Quick start

Run everything from this directory.

```bash
pip install -r requirements.txt

# live steering: recorded drive if present, else camera 0
python live.py
python live.py --source path/to/drive.mp4 --model b2   # other video, heavier model
```

### Working with a recorded drive

Put a recorded drive in `data/Caddy-Training-Data-2026-05-03_16-08-00/`, or set
`SEGDRIVE_DATA_DIR` to its folder. A drive folder holds `front-wide.mp4`,
`front-narrow.mp4`, `gps.json`, `ego.jsonl`, `control.jsonl`, and `map_route.json`.

```bash
modal run cloud/eval_segmentation.py       # segment on a cloud GPU → outputs/eval/
python tools/render_segmentation.py        # overlay + BEV videos
python tools/render_trajectories.py        # planner + steering video
python live.py                             # with no --source, replays the drive with its map route
```

### On the cart

```bash
# build a TensorRT engine for SegFormer
python onboard/export_segformer_trt.py

# 1. planner sidecar: reads the cameras, writes frames to /tmp/cart_frames/
#    and steer + pedal targets to /tmp/autoware_state.json
python onboard/segmentation_infer.py

# 2. in a second terminal, with the wheels straight: preview, then drive
python onboard/drive.py
python onboard/drive.py --i-understand-this-drives
```

`drive.py` sends the sidecar's targets to the cart through
[`cart-api`](../../cart-api). Gas is capped at `FSD_GAS_LIMIT` (`--max-gas` to
change). If the sidecar stops publishing, `drive.py` releases the gas
immediately and brakes after 1 s. Ctrl-C brakes, then disarms. There is no
software takeover, so keep a hand on the hardware e-stop.
