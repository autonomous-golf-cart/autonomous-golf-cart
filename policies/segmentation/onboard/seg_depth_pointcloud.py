#!/usr/bin/env python3
"""
Build a metric point cloud by fusing depth with semantic segmentation.

The main offline path this supports is the drive-by-segmentation output:

    scripts/seg_depth_pointcloud.py \
        --depth /path/to/depth_maps.npz \
        --seg /path/to/seg_maps_30fps.npz \
        --calib policies/segmentation/calibration/camera_calibration.json \
        --frame 0 \
        --output occupied_cloud.ply

Depth is unprojected with pinhole intrinsics into camera coordinates:
X right, Y down, Z forward, all in meters. The PLY colors come from the
segmentation class map using the Cityscapes palette by default.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CITYSCAPES_COLORS_RGB = np.array([
    (128, 64, 128),   # 0 road
    (244, 35, 232),   # 1 sidewalk
    (70, 70, 70),     # 2 building
    (102, 102, 156),  # 3 wall
    (190, 153, 153),  # 4 fence
    (153, 153, 153),  # 5 pole
    (250, 170, 30),   # 6 traffic light
    (220, 220, 0),    # 7 traffic sign
    (107, 142, 35),   # 8 vegetation
    (152, 251, 152),  # 9 terrain
    (70, 130, 180),   # 10 sky
    (220, 20, 60),    # 11 person
    (255, 0, 0),      # 12 rider
    (0, 0, 142),      # 13 car
    (0, 0, 70),       # 14 truck
    (0, 60, 100),     # 15 bus
    (0, 80, 100),     # 16 train
    (0, 0, 230),      # 17 motorcycle
    (119, 11, 32),    # 18 bicycle
], dtype=np.uint8)

SEGFORMER_MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

DEFAULT_LIVE_OUTPUT = Path("/tmp/cart_pointcloud/occupied_cloud.ply")
DEFAULT_LIVE_STATE = Path("/tmp/cart_pointcloud/state.json")
DEFAULT_CALIB = Path(__file__).resolve().parent.parent / "calibration" / "camera_calibration.json"

CLASS_ALIASES = {
    # Physical scene classes that should generally block free space.
    "occupied": (2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18),
    # Dynamic / road-user obstacles only.
    "obstacles": (11, 12, 13, 14, 15, 16, 17, 18),
    # Every semantic class with valid depth, including free ground.
    "all": tuple(range(len(CITYSCAPES_COLORS_RGB))),
}


def parse_class_filter(value: str) -> set[int]:
    value = value.strip().lower()
    if value in CLASS_ALIASES:
        return set(CLASS_ALIASES[value])
    try:
        return {int(part) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        aliases = ", ".join(sorted(CLASS_ALIASES))
        raise argparse.ArgumentTypeError(
            f"classes must be one of {aliases} or a comma-separated id list"
        ) from exc


def select_array(npz: np.lib.npyio.NpzFile, preferred: Iterable[str],
                 explicit_key: str | None, path: Path) -> tuple[np.ndarray, str]:
    if explicit_key:
        if explicit_key not in npz.files:
            raise KeyError(f"{path} does not contain key {explicit_key!r}; has {npz.files}")
        return npz[explicit_key], explicit_key
    for key in preferred:
        if key in npz.files:
            return npz[key], key
    if len(npz.files) == 1:
        return npz[npz.files[0]], npz.files[0]
    raise KeyError(f"could not choose array in {path}; pass --*-key. Keys: {npz.files}")


def frame_slice(array: np.ndarray, frame: int, name: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if not 0 <= frame < array.shape[0]:
            raise IndexError(f"--frame {frame} is outside {name} frame range 0..{array.shape[0] - 1}")
        return array[frame]
    raise ValueError(f"{name} must be 2D or frame-major 3D, got shape {array.shape}")


def load_depth_m(args: argparse.Namespace) -> np.ndarray:
    path = Path(args.depth)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path) as data:
            raw_all, key = select_array(
                data,
                ("depth_maps", "depth_map", "depth_m", "depth_mm", "depth", "arr_0"),
                args.depth_key,
                path,
            )
            raw = frame_slice(raw_all, args.frame, "depth").astype(np.float32)
            if "depth_scales" in data:
                scales = np.asarray(data["depth_scales"], dtype=np.float32).reshape(-1)
                scale = float(scales[min(args.frame, len(scales) - 1)])
                return raw / 65535.0 * scale
            if key == "depth_m":
                return raw
            return convert_depth_units(raw, args)

    if suffix == ".npy":
        raw = frame_slice(np.load(path), args.frame, "depth")
        return convert_depth_units(raw, args)

    raw_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise FileNotFoundError(f"could not read depth image: {path}")
    if raw_img.ndim == 3:
        raw_img = raw_img[:, :, 0]
    return convert_depth_units(raw_img, args)


def convert_depth_units(raw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.depth_max_m is not None and raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0 * float(args.depth_max_m)
    return raw.astype(np.float32) * float(args.depth_unit)


def color_image_to_class_map(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    flat = rgb.reshape(-1, 3).astype(np.int16)
    palette = CITYSCAPES_COLORS_RGB.astype(np.int16)
    d2 = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(rgb.shape[:2]).astype(np.uint8)


def load_segmentation(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(args.seg)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path) as data:
            raw_all, _ = select_array(
                data,
                ("seg_maps", "seg_map", "segmentation", "seg", "labels", "arr_0"),
                args.seg_key,
                path,
            )
            raw = frame_slice(raw_all, args.frame, "segmentation")
    elif suffix == ".npy":
        raw = frame_slice(np.load(path), args.frame, "segmentation")
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"could not read segmentation image: {path}")
        if img.ndim == 2:
            raw = img
        else:
            class_map = color_image_to_class_map(img[:, :, :3])
            colors = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            return class_map, colors

    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        rgb = raw[:, :, :3].astype(np.uint8)
        return color_image_to_class_map(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), rgb
    if raw.ndim != 2:
        raise ValueError(f"segmentation must be 2D labels or RGB image, got shape {raw.shape}")
    return raw.astype(np.int32), None


def load_intrinsics(args: argparse.Namespace, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    fx = fy = cx = cy = None
    calib_w = calib_h = None

    if args.calib:
        with Path(args.calib).open() as f:
            calib = json.load(f)
        intr = calib.get("intrinsics", calib)
        resolution = intr.get("resolution", calib.get("resolution"))
        fx = intr.get("fx", intr.get("focal_length", intr.get("focal")))
        fy = intr.get("fy", fx)
        cx = intr.get("cx", intr.get("principal_x"))
        cy = intr.get("cy", intr.get("principal_y"))
        calib_w = intr.get("width", calib.get("width", calib.get("image_width")))
        calib_h = intr.get("height", calib.get("height", calib.get("image_height")))
        if resolution and len(resolution) >= 2:
            calib_w = calib_w if calib_w is not None else resolution[0]
            calib_h = calib_h if calib_h is not None else resolution[1]

    fx = args.fx if args.fx is not None else fx
    if fx is None:
        raise ValueError("missing fx; pass --calib or --fx")
    fy = args.fy if args.fy is not None else (fy if fy is not None else fx)
    fx = float(fx)
    fy = float(fy)
    h, w = image_shape
    cx = float(args.cx if args.cx is not None else (cx if cx is not None else w / 2.0))
    cy = float(args.cy if args.cy is not None else (cy if cy is not None else h / 2.0))

    if calib_w and calib_h and (int(calib_w) != w or int(calib_h) != h):
        sx = w / float(calib_w)
        sy = h / float(calib_h)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    return fx, fy, cx, cy


def resize_seg_to_depth(seg: np.ndarray, colors: np.ndarray | None,
                        depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    dh, dw = depth_shape
    if seg.shape != depth_shape:
        seg = cv2.resize(seg.astype(np.int32), (dw, dh), interpolation=cv2.INTER_NEAREST)
    if colors is not None and colors.shape[:2] != depth_shape:
        colors = cv2.resize(colors, (dw, dh), interpolation=cv2.INTER_NEAREST)
    return seg, colors


def build_cloud(depth_m: np.ndarray, seg: np.ndarray, seg_rgb: np.ndarray | None,
                classes: set[int], intrinsics: tuple[float, float, float, float],
                args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    h, w = depth_m.shape
    finite_depth = np.isfinite(depth_m)
    depth_mask = finite_depth & (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m)
    class_mask = np.isin(seg, list(classes))
    mask = depth_mask & class_mask

    if args.stride > 1:
        stride_mask = np.zeros_like(mask)
        stride_mask[::args.stride, ::args.stride] = True
        mask &= stride_mask

    v, u = np.nonzero(mask)
    z = depth_m[v, u].astype(np.float32)
    x = ((u.astype(np.float32) - cx) * z / fx).astype(np.float32)
    y = ((v.astype(np.float32) - cy) * z / fy).astype(np.float32)
    points = np.column_stack((x, y, z)).astype(np.float32)

    if seg_rgb is not None:
        colors = seg_rgb[v, u, :3].astype(np.uint8)
    else:
        clamped = np.clip(seg[v, u], 0, len(CITYSCAPES_COLORS_RGB) - 1)
        colors = CITYSCAPES_COLORS_RGB[clamped].astype(np.uint8)
    return points, colors


def build_full_seg_cloud(depth_m: np.ndarray, seg: np.ndarray,
                         intrinsics: tuple[float, float, float, float],
                         args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    h, w = depth_m.shape
    finite = np.isfinite(depth_m)
    mask = finite & (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m)
    if args.stride > 1:
        sm = np.zeros_like(mask)
        sm[::args.stride, ::args.stride] = True
        mask &= sm
    v, u = np.nonzero(mask)
    if v.size == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8))
    z = depth_m[v, u].astype(np.float32)
    x = ((u.astype(np.float32) - cx) * z / fx).astype(np.float32)
    y = ((v.astype(np.float32) - cy) * z / fy).astype(np.float32)
    points = np.column_stack((x, y, z)).astype(np.float32)
    if seg.shape != (h, w):
        seg = cv2.resize(seg.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    cls = np.clip(seg[v, u], 0, len(CITYSCAPES_COLORS_RGB) - 1)
    colors_rgb = CITYSCAPES_COLORS_RGB[cls].astype(np.uint8)
    return points, colors_rgb


def create_seg_overlay(frame_bgr: np.ndarray, seg: np.ndarray,
                       alpha: float = 0.45) -> np.ndarray:
    seg_resized = seg
    h, w = frame_bgr.shape[:2]
    if seg_resized.shape != (h, w):
        seg_resized = cv2.resize(seg_resized.astype(np.int32), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
    rgb = CITYSCAPES_COLORS_RGB[
        np.clip(seg_resized, 0, len(CITYSCAPES_COLORS_RGB) - 1)
    ]
    color_bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, color_bgr, alpha, 0.0)


def create_depth_viz(depth_m: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    finite = np.isfinite(depth_m)
    valid = finite & (depth_m > 0)
    if np.any(valid):
        hi = float(np.percentile(depth_m[valid], 98.0))
        hi = max(hi, 1.0)
        norm = np.clip(depth_m / hi * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth_m, dtype=np.uint8)
    bgr = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    return cv2.resize(bgr, out_size, interpolation=cv2.INTER_LINEAR)


def _nice_grid_step(extent_m: float) -> float:
    for step in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        if extent_m / step <= 10.0:
            return step
    return 100.0


def _auto_bev_range(points: np.ndarray, min_forward: float,
                    min_lateral: float, pad: float = 0.10) -> tuple[float, float]:
    if points.size == 0:
        return min_forward, min_lateral
    z = points[:, 2]
    x = points[:, 0]
    z_pos = z[z > 0]
    forward = float(z_pos.max()) if z_pos.size else min_forward
    lateral = float(np.abs(x).max()) if x.size else min_lateral
    forward = max(min_forward, forward * (1.0 + pad))
    lateral = max(min_lateral, lateral * (1.0 + pad))
    return forward, lateral


def create_cloud_bev(points: np.ndarray, colors_rgb: np.ndarray,
                     size: int, forward_m: float, lateral_m: float,
                     auto: bool = True) -> np.ndarray:
    bev = np.full((size, size, 3), (20, 22, 24), dtype=np.uint8)
    if points.size == 0:
        cv2.putText(bev, "no occupied points", (18, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        return bev

    if auto:
        forward_m, lateral_m = _auto_bev_range(
            points, min_forward=2.0, min_lateral=1.0,
        )

    x = points[:, 0]
    z = points[:, 2]
    keep = (z > 0) & (z <= forward_m) & (np.abs(x) <= lateral_m)
    if np.any(keep):
        x = x[keep]
        z = z[keep]
        colors_bgr = colors_rgb[keep][:, ::-1]
        px = ((x / lateral_m * 0.5 + 0.5) * (size - 1)).astype(np.int32)
        py = ((1.0 - z / forward_m) * (size - 1)).astype(np.int32)
        order = np.argsort(z)[::-1]
        bev[py[order], px[order]] = colors_bgr[order]

    fwd_step = _nice_grid_step(forward_m)
    lat_step = _nice_grid_step(lateral_m * 2.0)
    dist = fwd_step
    while dist <= forward_m + 1e-6:
        y = int((1.0 - dist / forward_m) * (size - 1))
        cv2.line(bev, (0, y), (size - 1, y), (48, 52, 57), 1)
        label = f"{dist:g}m"
        cv2.putText(bev, label, (6, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 154, 160), 1, cv2.LINE_AA)
        dist += fwd_step
    offset = -lateral_m + lat_step
    while offset < lateral_m - 1e-6:
        if abs(offset) > 1e-6:
            x_px = int((offset / lateral_m * 0.5 + 0.5) * (size - 1))
            cv2.line(bev, (x_px, 0), (x_px, size - 1), (40, 44, 49), 1)
        offset += lat_step

    ego = np.array([[size // 2, size - 10], [size // 2 - 9, size - 28],
                    [size // 2 + 9, size - 28]], dtype=np.int32)
    cv2.fillPoly(bev, [ego], (245, 245, 245))
    cv2.putText(bev, "top-down seg cloud", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(bev, f"{forward_m:.1f}m x {lateral_m * 2:.1f}m",
                (12, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (190, 190, 196), 1, cv2.LINE_AA)
    return bev


def create_live_viz(frame_bgr: np.ndarray, depth_m: np.ndarray, seg: np.ndarray,
                    points: np.ndarray, colors_rgb: np.ndarray,
                    frame_idx: int, elapsed_s: float, args: argparse.Namespace,
                    bev_points: np.ndarray | None = None,
                    bev_colors_rgb: np.ndarray | None = None) -> np.ndarray:
    panel_w, panel_h = 480, 360
    overlay = create_seg_overlay(frame_bgr, seg)
    overlay = cv2.resize(overlay, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    depth = create_depth_viz(depth_m, (panel_w, panel_h))
    bev = create_cloud_bev(
        bev_points if bev_points is not None else points,
        bev_colors_rgb if bev_colors_rgb is not None else colors_rgb,
        size=panel_h,
        forward_m=args.viz_forward_m,
        lateral_m=args.viz_lateral_m,
        auto=not args.bev_fixed_range,
    )
    bev = cv2.resize(bev, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((panel_h + 38, panel_w * 3, 3), dtype=np.uint8)
    canvas[:panel_h, :panel_w] = overlay
    canvas[:panel_h, panel_w:panel_w * 2] = depth
    canvas[:panel_h, panel_w * 2:] = bev
    cv2.line(canvas, (panel_w, 0), (panel_w, panel_h), (70, 70, 70), 1)
    cv2.line(canvas, (panel_w * 2, 0), (panel_w * 2, panel_h), (70, 70, 70), 1)

    cv2.putText(canvas, "camera + segmentation", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "metric depth", (panel_w + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "point cloud BEV", (panel_w * 2 + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    fps = 1.0 / elapsed_s if elapsed_s > 0 else 0.0
    status = (
        f"frame {frame_idx}  points {len(points):,}  infer {elapsed_s:.2f}s  "
        f"{fps:.1f} fps  press q/esc to close"
    )
    cv2.putText(canvas, status, (12, panel_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 224, 230), 1, cv2.LINE_AA)
    return canvas


class VtkPointCloudViewer:
    """Non-blocking VTK viewer with native mouse rotate/pan/zoom controls."""

    def __init__(self, title: str, point_size: float):
        import vtk

        self.vtk = vtk
        self.points = vtk.vtkPoints()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("colors")

        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)
        self.poly.GetPointData().SetScalars(self.colors)

        self.glyph = vtk.vtkVertexGlyphFilter()
        self.glyph.SetInputData(self.poly)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.03, 0.035, 0.04)
        self.renderer.AddActor(actor)
        self.renderer.AddActor(axes)

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName(title)
        self.window.SetSize(1100, 760)
        self.window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0.0, 3.0, -8.0)
        camera.SetFocalPoint(0.0, 0.0, 8.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        self._first_update = True

    def update(self, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
        vtk = self.vtk
        self.points.Reset()
        self.colors.Reset()

        if len(points_xyz):
            # Point-cloud file coordinates are X right, Y down, Z forward.
            # The viewer uses Y up so the live cloud is naturally oriented.
            viewer_xyz = points_xyz.copy()
            viewer_xyz[:, 1] *= -1.0
            for (x, y, z), (r, g, b) in zip(viewer_xyz, colors_rgb):
                self.points.InsertNextPoint(float(x), float(y), float(z))
                self.colors.InsertNextTypedTuple((int(r), int(g), int(b)))

        self.points.Modified()
        self.colors.Modified()
        self.poly.Modified()
        self.glyph.Update()
        if self._first_update:
            self.renderer.ResetCamera()
            self.renderer.GetActiveCamera().Azimuth(180)
            self.renderer.GetActiveCamera().Elevation(15)
            self.renderer.ResetCameraClippingRange()
            self._first_update = False
        self.window.Render()
        self.interactor.ProcessEvents()

    def close(self) -> None:
        self.window.Finalize()
        self.interactor.TerminateApp()


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray,
              binary: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        "ply\n"
        f"format {fmt}\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    mode = "wb" if binary else "w"
    with path.open(mode) as f:
        if binary:
            f.write(header.encode("ascii"))
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(struct.pack("<fffBBB", float(x), float(y), float(z),
                                    int(r), int(g), int(b)))
        else:
            f.write(header)
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_ply_atomic(path: Path, points: np.ndarray, colors: np.ndarray,
                     binary: bool = True) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_ply(tmp, points, colors, binary=binary)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, path)


def discover_v4l2_indices(count: int = 1, max_scan: int = 16) -> list[int]:
    found: list[int] = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
            if len(found) >= count:
                break
    return found


class LiveFrameSource:
    def __init__(self, source: str, width: int, height: int,
                 max_scan: int = 16, loop_video: bool = True,
                 flip_code: int | None = None):
        self.source = source
        self.width = width
        self.height = height
        self.loop_video = loop_video
        self.flip_code = flip_code
        self.image_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None

        if source == "auto":
            indices = discover_v4l2_indices(count=1, max_scan=max_scan)
            if not indices:
                raise RuntimeError(f"no v4l2 camera found while scanning 0..{max_scan - 1}")
            self.open_capture(indices[0], use_v4l2=True)
            self.label = f"/dev/video{indices[0]}"
        elif source.isdigit():
            self.open_capture(int(source), use_v4l2=True)
            self.label = f"/dev/video{source}"
        else:
            path = Path(source)
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                self.image_path = path
                self.label = str(path)
            else:
                self.open_capture(str(path), use_v4l2=False)
                self.label = str(path)

    def open_capture(self, source: int | str, use_v4l2: bool) -> None:
        api = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, api)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"failed to open live source {source}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

    def read(self) -> np.ndarray | None:
        if self.image_path is not None:
            frame = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                return None
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)
            return frame

        assert self.cap is not None
        ok, frame = self.cap.read()
        if (not ok or frame is None) and self.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.flip_code is not None:
            frame = cv2.flip(frame, self.flip_code)
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class LiveSegDepthPipeline:
    def __init__(self, device: str | None, seg_model: str,
                 depth_model: str | None):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForDepthEstimation,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        seg_name = SEGFORMER_MODELS.get(seg_model, seg_model)
        print(f"[models] loading segmentation {seg_name} on {device}", flush=True)
        self.seg_processor = SegformerImageProcessor.from_pretrained(seg_name)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_name)
        self.seg_model = self.seg_model.to(device).eval()

        self.depth_processor = None
        self.depth_model = None
        if depth_model:
            print(f"[models] loading depth {depth_model} on {device}", flush=True)
            self.depth_processor = AutoImageProcessor.from_pretrained(depth_model)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                depth_model, torch_dtype=torch_dtype,
            ).to(device).eval()

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        target_size = rgb.shape[:2]

        seg_inputs = self.seg_processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            seg_out = self.seg_model(**seg_inputs)
        return self.seg_processor.post_process_semantic_segmentation(
            seg_out, target_sizes=[target_size],
        )[0].detach().cpu().numpy().astype(np.uint8)

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image
        from torch.nn import functional as F

        if self.depth_model is None:
            raise RuntimeError("depth model not loaded; use segment() instead")

        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        target_size = rgb.shape[:2]

        seg_inputs = self.seg_processor(images=pil, return_tensors="pt").to(self.device)
        depth_inputs = self.depth_processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            seg_out = self.seg_model(**seg_inputs)
            depth_out = self.depth_model(**depth_inputs).predicted_depth

        seg = self.seg_processor.post_process_semantic_segmentation(
            seg_out, target_sizes=[target_size],
        )[0].detach().cpu().numpy().astype(np.uint8)

        depth = F.interpolate(
            depth_out.unsqueeze(1).float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return depth, seg


class TrtSegPipeline:
    """SegFormer inference via a TensorRT engine.

    Engine is expected to take an NCHW float32/float16 input
    (1, 3, H, W) with ImageNet normalization, and return logits
    (1, C, H/4, W/4). Argmax + nearest-neighbor upsample give the
    final class map at the camera frame resolution.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    def __init__(self, engine_path: str, device: str | None = None):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.device.startswith("cuda"):
            raise SystemExit("TensorRT seg backend requires a CUDA device.")

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise SystemExit(f"failed to deserialize TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name: str | None = None
        self.output_name: str | None = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(self.engine.get_tensor_shape(name))
                self.input_dtype = self._np_dtype(self.engine.get_tensor_dtype(name))
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
                self.output_shape = tuple(self.engine.get_tensor_shape(name))
                self.output_dtype = self._np_dtype(self.engine.get_tensor_dtype(name))
        if self.input_name is None or self.output_name is None:
            raise SystemExit("TRT engine missing input/output tensor")
        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise SystemExit(f"unexpected TRT input shape {self.input_shape}; expected (1,3,H,W)")
        _, _, self.in_h, self.in_w = self.input_shape

        torch_in_dtype = torch.float16 if self.input_dtype == np.float16 else torch.float32
        torch_out_dtype = torch.float16 if self.output_dtype == np.float16 else torch.float32
        self.in_buf = torch.empty(self.input_shape, dtype=torch_in_dtype,
                                  device=self.device).contiguous()
        self.out_buf = torch.empty(self.output_shape, dtype=torch_out_dtype,
                                   device=self.device).contiguous()
        self.context.set_tensor_address(self.input_name, self.in_buf.data_ptr())
        self.context.set_tensor_address(self.output_name, self.out_buf.data_ptr())
        self.stream = torch.cuda.Stream()

        mean = torch.from_numpy(self.IMAGENET_MEAN).to(self.device).view(1, 3, 1, 1)
        std = torch.from_numpy(self.IMAGENET_STD).to(self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

        print(
            f"[trt] engine={engine_path} input={self.input_name}{self.input_shape}"
            f" ({self.input_dtype.__name__}) "
            f"output={self.output_name}{self.output_shape} ({self.output_dtype.__name__})",
            flush=True,
        )

    @staticmethod
    def _np_dtype(trt_dtype) -> type:
        import tensorrt as trt
        return {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
        }[trt_dtype]

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (self.in_w, self.in_h),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        tensor = (tensor - self._mean) / self._std
        self.in_buf.copy_(tensor.to(self.in_buf.dtype))

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TRT execute_async_v3 returned False")
        self.stream.synchronize()

        logits = self.out_buf
        cls = logits.argmax(dim=1)[0]
        cls_cpu = cls.to(torch.uint8).cpu().numpy()
        if cls_cpu.shape != (h, w):
            cls_cpu = cv2.resize(cls_cpu, (w, h), interpolation=cv2.INTER_NEAREST)
        return cls_cpu


class RealSenseFrameSource:
    def __init__(self, width: int, height: int, fps: int = 30,
                 flip_code: int | None = None):
        import pyrealsense2 as rs

        self.rs = rs
        self.width = width
        self.height = height
        self.flip_code = flip_code
        self.label = "realsense"

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)

        device = profile.get_device()
        self.label = (
            f"realsense:{device.get_info(rs.camera_info.name)}"
            f"#{device.get_info(rs.camera_info.serial_number)}"
        )

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = (float(intr.fx), float(intr.fy),
                           float(intr.ppx), float(intr.ppy))

        self.align = rs.align(rs.stream.color)
        print(
            f"[realsense] {self.label} {width}x{height}@{fps} "
            f"fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f} "
            f"depth_scale={self.depth_scale}",
            flush=True,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        frames = self.pipeline.wait_for_frames(5000)
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None
        color_bgr = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        if self.flip_code is not None:
            color_bgr = cv2.flip(color_bgr, self.flip_code)
            depth_m = cv2.flip(depth_m, self.flip_code)
        return color_bgr, depth_m

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def run_live(args: argparse.Namespace) -> int:
    use_realsense = args.source == "realsense"
    if not use_realsense and args.calib is None and args.fx is None:
        if DEFAULT_CALIB.exists():
            args.calib = str(DEFAULT_CALIB)
        else:
            raise SystemExit("Live mode needs --calib or --fx for metric point sizing.")

    realsense_source: RealSenseFrameSource | None = None
    source: LiveFrameSource | RealSenseFrameSource
    if use_realsense:
        realsense_source = RealSenseFrameSource(
            width=args.live_width,
            height=args.live_height,
            fps=args.realsense_fps,
            flip_code=args.flip_code,
        )
        source = realsense_source
    else:
        source = LiveFrameSource(
            args.source,
            width=args.live_width,
            height=args.live_height,
            max_scan=args.max_scan,
            loop_video=not args.no_loop,
            flip_code=args.flip_code,
        )
    if args.seg_backend == "trt":
        if not args.trt_engine:
            raise SystemExit("--seg-backend trt requires --trt-engine PATH")
        pipeline = TrtSegPipeline(args.trt_engine, device=args.device)
        if not use_realsense:
            raise SystemExit("--seg-backend trt currently only supports --source realsense")
    else:
        pipeline = LiveSegDepthPipeline(
            args.device, args.seg_model,
            None if use_realsense else args.depth_model,
        )
    cloud_viewer = None
    if args.show_window:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, 1440, 398)
    if args.show_3d_window:
        cloud_viewer = VtkPointCloudViewer(args.cloud_window_name, args.point_size)

    period = 1.0 / max(args.live_hz, 1e-3)
    frames = 0
    print(
        f"[live] source={source.label} output={args.output} "
        f"classes={sorted(args.classes)} hz={args.live_hz:g}",
        flush=True,
    )

    try:
        while True:
            started = time.perf_counter()
            stages: dict[str, float] = {}

            def stage(name: str, t0: float) -> float:
                t1 = time.perf_counter()
                stages[name] = (t1 - t0) * 1000.0
                return t1

            t = started
            if use_realsense:
                rs_out = realsense_source.read()
                if rs_out is None:
                    frame = None
                    depth_m = None
                else:
                    frame, depth_m = rs_out
                    depth_m = np.where(
                        (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m),
                        depth_m, 0.0,
                    ).astype(np.float32)
            else:
                frame = source.read()
                depth_m = None
            t = stage("capture", t)
            if frame is None:
                write_json_atomic(args.state_file, {
                    "ok": False,
                    "error": f"no frame from {source.label}",
                    "source": source.label,
                    "ts": time.time(),
                })
                time.sleep(min(period, 0.5))
                continue

            if use_realsense:
                seg = pipeline.segment(frame)
                intrinsics = realsense_source.intrinsics
                if depth_m.shape != seg.shape:
                    seg = cv2.resize(seg, (depth_m.shape[1], depth_m.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                depth_m, seg = pipeline(frame)
                intrinsics = load_intrinsics(args, depth_m.shape)
            t = stage("infer", t)

            seg, seg_rgb = resize_seg_to_depth(seg, None, depth_m.shape)
            points, colors = build_cloud(depth_m, seg, seg_rgb, args.classes, intrinsics, args)
            t = stage("build_cloud", t)

            if args.write_ply:
                write_ply_atomic(args.output, points, colors, binary=not args.ascii)
                t = stage("ply_write", t)

            frames += 1
            elapsed_pre_viz = time.perf_counter() - started

            if args.show_window:
                bev_points, bev_colors = build_full_seg_cloud(
                    depth_m, seg, intrinsics, args,
                )
                t = stage("viz_build_cloud", t)
                viz = create_live_viz(frame, depth_m, seg, points, colors,
                                      frames, elapsed_pre_viz, args,
                                      bev_points=bev_points,
                                      bev_colors_rgb=bev_colors)
                cv2.imshow(args.window_name, viz)
                key = cv2.waitKey(1) & 0xFF
                t = stage("viz_render", t)
                if key in (27, ord("q")):
                    print("[live] visualization window closed by user", flush=True)
                    break
            if cloud_viewer is not None:
                cloud_viewer.update(points, colors)
                t = stage("vtk_update", t)

            elapsed = time.perf_counter() - started
            write_json_atomic(args.state_file, {
                "ok": True,
                "source": source.label,
                "output": str(args.output),
                "points": int(len(points)),
                "frame": frames,
                "infer_s": round(elapsed, 4),
                "fps": round(1.0 / elapsed, 3) if elapsed > 0 else 0.0,
                "classes": sorted(int(c) for c in args.classes),
                "ts": time.time(),
            })

            print(
                f"[live] f={frames} pts={len(points):,} "
                f"dt={elapsed*1000:.0f}ms ({1.0/elapsed:.1f}fps) "
                + " ".join(f"{k}={v:.0f}" for k, v in stages.items()),
                flush=True,
            )

            sleep_s = period - (time.perf_counter() - started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        source.close()
        if args.show_window:
            cv2.destroyAllWindows()
        if cloud_viewer is not None:
            cloud_viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse depth and segmentation into a class-colored occupied-pixel point cloud."
    )
    p.add_argument("--live", action="store_true",
                   help="Continuously infer depth+segmentation from a live source and rewrite the PLY.")
    p.add_argument("--depth", default=None, help="Depth .npz/.npy/image path.")
    p.add_argument("--seg", default=None, help="Segmentation .npz/.npy/image path.")
    p.add_argument("--output", default=DEFAULT_LIVE_OUTPUT, type=Path, help="Output .ply path.")
    p.add_argument("--frame", type=int, default=0, help="Frame index for frame-major arrays.")
    p.add_argument("--depth-key", default=None, help="Array key inside depth .npz.")
    p.add_argument("--seg-key", default=None, help="Array key inside segmentation .npz.")
    p.add_argument("--calib", default=None, help="Calibration JSON with intrinsics.")
    p.add_argument("--fx", type=float, default=None, help="Focal length x in pixels.")
    p.add_argument("--fy", type=float, default=None, help="Focal length y in pixels.")
    p.add_argument("--cx", type=float, default=None, help="Principal point x in pixels.")
    p.add_argument("--cy", type=float, default=None, help="Principal point y in pixels.")
    p.add_argument("--depth-unit", type=float, default=0.001,
                   help="Meters per raw depth unit when no depth_scales exists (default: mm).")
    p.add_argument("--depth-max-m", type=float, default=None,
                   help="For 8-bit normalized depth images, map 255 to this many meters.")
    p.add_argument("--min-depth-m", type=float, default=0.3)
    p.add_argument("--max-depth-m", type=float, default=80.0)
    p.add_argument("--classes", type=parse_class_filter, default=parse_class_filter("occupied"),
                   help="occupied, obstacles, all, or comma-separated class ids. Default: occupied.")
    p.add_argument("--stride", type=int, default=1,
                   help="Keep one pixel every N rows/cols to thin large clouds.")
    p.add_argument("--ascii", action="store_true", help="Write ASCII PLY instead of binary PLY.")
    p.add_argument("--source", default="auto",
                   help="Live source: auto, v4l2 index, video path, image path, "
                        "or 'realsense' to use D4xx color+depth.")
    p.add_argument("--realsense-fps", type=int, default=30,
                   help="Frame rate for --source realsense color+depth streams.")
    p.add_argument("--live-width", type=int, default=640,
                   help="Live frame width after capture/resize.")
    p.add_argument("--live-height", type=int, default=480,
                   help="Live frame height after capture/resize.")
    p.add_argument("--live-hz", type=float, default=1.0,
                   help="Target live point-cloud publish rate.")
    p.add_argument("--state-file", default=DEFAULT_LIVE_STATE, type=Path,
                   help="Live JSON status path.")
    p.add_argument("--seg-model", default="b0", choices=tuple(SEGFORMER_MODELS),
                   help="SegFormer variant for live segmentation (Hugging Face backend).")
    p.add_argument("--seg-backend", default="hf", choices=("hf", "trt"),
                   help="Segmentation backend: 'hf' (PyTorch+transformers) or 'trt' (TensorRT engine).")
    p.add_argument("--trt-engine", default=None,
                   help="Path to a TensorRT engine file (used when --seg-backend trt).")
    p.add_argument("--depth-model",
                   default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
                   help="Hugging Face metric depth model for live depth.")
    p.add_argument("--device", default=None, help="Torch device for live models.")
    p.add_argument("--max-scan", type=int, default=16,
                   help="Highest v4l2 index scan bound for --source auto.")
    p.add_argument("--no-loop", action="store_true",
                   help="With a live video source, stop at EOF instead of looping.")
    p.add_argument("--show-window", action="store_true",
                   help="Show a live OpenCV visualization window.")
    p.add_argument("--show-3d-window", action="store_true",
                   help="Show a mouse-interactive VTK 3D point-cloud window.")
    p.add_argument("--window-name", default="Segmentation + Depth Point Cloud",
                   help="OpenCV visualization window title.")
    p.add_argument("--cloud-window-name", default="Live Occupied Point Cloud",
                   help="VTK 3D point-cloud window title.")
    p.add_argument("--write-ply", action="store_true",
                   help="Write the occupied PLY to --output each frame (off by default in live mode for speed).")
    p.add_argument("--bev-fixed-range", action="store_true",
                   help="Disable BEV autoscale and use --viz-forward-m / --viz-lateral-m as a fixed window.")
    p.add_argument("--viz-forward-m", type=float, default=25.0,
                   help="Forward range shown in the top-down point-cloud panel.")
    p.add_argument("--viz-lateral-m", type=float, default=12.5,
                   help="Half-width shown in the top-down point-cloud panel.")
    p.add_argument("--point-size", type=float, default=2.0,
                   help="Rendered point size for the interactive 3D cloud.")
    p.add_argument("--flip", default="none", choices=("none", "0", "1", "-1"),
                   help="Flip live frames before inference/viz. -1 is 180 degrees.")
    args = p.parse_args()
    args.flip_code = None if args.flip == "none" else int(args.flip)
    return args


def main() -> int:
    args = parse_args()
    if args.live:
        return run_live(args)

    if args.depth is None or args.seg is None:
        raise SystemExit("Offline mode needs --depth and --seg. Use --live for camera inference.")
    if args.fx is None and not args.calib:
        raise SystemExit("Pass --calib or --fx. Intrinsics are required for metric sizing.")

    depth_m = load_depth_m(args)
    seg, seg_rgb = load_segmentation(args)
    seg, seg_rgb = resize_seg_to_depth(seg, seg_rgb, depth_m.shape)
    intrinsics = load_intrinsics(args, depth_m.shape)

    points, colors = build_cloud(depth_m, seg, seg_rgb, args.classes, intrinsics, args)
    write_ply(args.output, points, colors, binary=not args.ascii)

    fx, fy, cx, cy = intrinsics
    print(
        f"Wrote {len(points):,} points to {args.output} "
        f"(fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
