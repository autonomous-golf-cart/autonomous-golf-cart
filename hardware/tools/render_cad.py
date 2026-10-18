#!/usr/bin/env python3
"""Render the original STL geometry as shaded, orthographic CAD previews.

Run from any directory:
    python3 hardware/tools/render_cad.py
Requires numpy and Pillow. No CAD application, GPU, or network is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

HARDWARE = Path(__file__).resolve().parents[1]
STL_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
GROUP_NAMES = {
    "pedals": "Pedal actuation",
    "steering": "Steering & belt drive",
    "cameras": "Camera rig",
    "sensors": "Sensor housings",
    "accessories": "Display & connectivity",
}
PALETTE = {
    "background": "#ffffff",
    "foreground": "#26251d",
    "muted": "#6b6756",
    "secondary": "#f7f7f4",
    "border": "#e5e3dc",
    "model": (168, 163, 144),
}


def read_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    vertices = np.frombuffer(data, dtype=STL_RECORD, count=count, offset=84)["vertices"]
    if not np.isfinite(vertices).all():
        raise ValueError(f"Nonfinite coordinates: {path}")
    return vertices.astype(np.float64)


def render(vertices, destination, width=1200, height=900):
    # A depth buffer prevents hidden triangles showing through flat faces.
    supersample = 2
    w, h = width * supersample, height * supersample
    azimuth, elevation = math.radians(-57), math.radians(29)
    camera = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth), math.sin(elevation)
    ])
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0])
    up = np.cross(camera, right)
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    projected = vertices @ np.stack((right, up, camera), axis=1)
    lo, hi = projected.min(axis=(0, 1)), projected.max(axis=(0, 1))
    scale = min(w * 0.84 / max(hi[0] - lo[0], 1e-9), h * 0.80 / max(hi[1] - lo[1], 1e-9))
    projected[:, :, 0] = (projected[:, :, 0] - (lo[0] + hi[0]) / 2) * scale + w / 2
    projected[:, :, 1] = -(projected[:, :, 1] - (lo[1] + hi[1]) / 2) * scale + h / 2
    depth = np.full((h, w), -np.inf, dtype=np.float32)
    face_index = np.full((h, w), -1, dtype=np.int32)
    light = np.array([-0.3, -0.5, 0.82])
    light /= np.linalg.norm(light)
    intensity = 0.63 + 0.35 * np.maximum(normals @ light, 0)
    colors = np.clip(intensity[:, None] * np.array(PALETTE["model"]), 0, 255).astype(np.uint8)

    # Rasterize both sides: STL exports may contain inconsistent winding.
    for i, triangle in enumerate(projected):
        x0 = max(0, math.floor(triangle[:, 0].min()))
        x1 = min(w - 1, math.ceil(triangle[:, 0].max()))
        y0 = max(0, math.floor(triangle[:, 1].min()))
        y1 = min(h - 1, math.ceil(triangle[:, 1].max()))
        if x1 < x0 or y1 < y0 or lengths[i] < 1e-12:
            continue
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        u = ((b[1] - c[1]) * (xx + 0.5 - c[0]) + (c[0] - b[0]) * (yy + 0.5 - c[1])) / denominator
        v = ((c[1] - a[1]) * (xx + 0.5 - c[0]) + (a[0] - c[0]) * (yy + 0.5 - c[1])) / denominator
        inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        z = u * a[2] + v * b[2] + (1 - u - v) * c[2]
        patch = depth[y0:y1 + 1, x0:x1 + 1]
        update = inside & (z > patch)
        patch[update] = z[update]
        face_index[y0:y1 + 1, x0:x1 + 1][update] = i

    occupied = face_index >= 0
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[occupied] = colors[face_index[occupied]]
    edges = np.zeros((h, w), dtype=bool)
    # Show silhouette and real creases, without drawing tessellation diagonals.
    for dy, dx in [(0, 1), (1, 0)]:
        first = (slice(0, h - dy or h), slice(0, w - dx or w))
        second = (slice(dy, h), slice(dx, w))
        ia, ib = face_index[first], face_index[second]
        both = (ia >= 0) & (ib >= 0) & (ia != ib)
        crease = np.zeros_like(both)
        na, nb = normals[ia[both]], normals[ib[both]]
        crease[both] = np.abs(np.sum(na * nb, axis=1)) < math.cos(math.radians(35))
        boundary = ((ia >= 0) != (ib >= 0)) | crease
        edges[first] |= boundary & (ia >= 0)
        edges[second] |= boundary & (ib >= 0)
    rgb[edges] = np.array([82, 79, 66])
    result = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, optimize=True)
    return result


def font(size, face="roman"):
    # Bundle the real typefaces so regenerated images keep the same typography.
    fonts = HARDWARE / "tools/fonts"
    if face in {"ui", "mono"}:
        path = fonts / "geist" / ("Geist-Regular.ttf" if face == "ui" else "GeistMono-Regular.ttf")
    else:
        path = fonts / "cmu" / f"cmu-serif-{face}.ttf"
    return ImageFont.truetype(str(path), size)


def tracked_text(draw, xy, text, size=14, spacing=1.3, fill=None):
    x, y = xy
    face = font(size, "mono")
    for letter in text:
        draw.text((x, y), letter, fill=fill or PALETTE["muted"], font=face)
        x += draw.textlength(letter, font=face) + spacing


def wrap_text(draw, text, face, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=face) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def make_sheet(parts, destination, columns, title, subtitle, cell_width=480):
    card_height = 364
    margin, gutter, header, footer = 84, 20, 340, 92
    rows = math.ceil(len(parts) / columns)
    width = 2 * margin + columns * cell_width + (columns - 1) * gutter
    height = header + rows * card_height + (rows - 1) * gutter + footer
    sheet = Image.new("RGB", (width, height), PALETTE["background"])
    draw = ImageDraw.Draw(sheet)
    ink, muted, rule = PALETTE["foreground"], PALETTE["muted"], PALETTE["border"]
    draw.rectangle((margin, 47, margin + 10, 57), fill=ink)
    draw.text((margin + 26, 34), "Autonomous golf cart", fill=ink, font=font(28))
    nav = "HARDWARE / MECHANICAL DESIGN"
    nav_width = sum(draw.textlength(c, font=font(14, "mono")) + 1.3 for c in nav)
    tracked_text(draw, (width - margin - nav_width, 43), nav)
    draw.line((margin, 91, width - margin, 91), fill=rule, width=2)
    draw.text((margin, 134), title, fill=ink, font=font(74, "bold"))
    draw.text((margin + 3, 236), subtitle, fill=muted, font=font(27, "italic"))
    draw.line((width - margin - 290, 145, width - margin - 290, 278), fill=rule, width=2)
    tracked_text(draw, (width - margin - 258, 153), "COMPONENTS")
    draw.text((width - margin - 258, 182), "33", fill=ink, font=font(65, "bold"))
    draw.text((width - margin - 166, 220), "/ 5 subsystems", fill=muted, font=font(20, "ui"))
    draw.line((margin, 304, width - margin, 304), fill=rule, width=2)
    for i, part in enumerate(parts):
        x = margin + (i % columns) * (cell_width + gutter)
        y = header + (i // columns) * (card_height + gutter)
        draw.rectangle((x, y, x + cell_width, y + card_height), outline=rule, width=2)
        draw.rectangle((x + 2, y + 2, x + cell_width - 2, y + 40), fill=PALETTE["secondary"])
        draw.line((x, y + 41, x + cell_width, y + 41), fill=rule, width=1)
        tracked_text(draw, (x + 20, y + 12), GROUP_NAMES[part["group"]].upper(), spacing=0.8)
        number = f"{i + 1:02d}"
        draw.text((x + cell_width - 48, y + 9), number, fill=muted, font=font(18, "mono"))
        title_font = font(29, "bold")
        lines = wrap_text(draw, part["name"], title_font, cell_width - 40)
        if len(lines) > 2:
            raise ValueError(f'Component title does not fit: {part["name"]}')
        for j, line in enumerate(lines):
            draw.text((x + 20, y + 56 + j * 32), line, fill=ink, font=title_font)
        preview = Image.open(HARDWARE / part["render"]).convert("RGB")
        bounds = ImageChops.difference(preview, Image.new("RGB", preview.size, "white")).getbbox()
        if bounds:
            preview = preview.crop(bounds)
        preview.thumbnail((cell_width - 64, 194), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + (cell_width - preview.width) // 2,
                             y + 119 + (194 - preview.height) // 2))
        draw.line((x, y + 328, x + cell_width, y + 328), fill=rule, width=1)
        dimensions = " × ".join(f"{d:.1f}".removesuffix(".0") for d in part["dimensions"])
        draw.text((x + 20, y + 338), dimensions + " mm*", fill=muted, font=font(18, "ui"))
        draw.text((x + cell_width - 56, y + 340), "STL", fill=muted, font=font(14, "mono"))
    bottom = height - footer + 30
    draw.line((margin, bottom, width - margin, bottom), fill=rule, width=2)
    draw.text((margin, bottom + 20), "Original mesh geometry · *STL dimensions interpreted in millimeters.", fill=muted, font=font(18, "ui"))
    note = "Individual views are not to a common scale."
    draw.text((width - margin - draw.textlength(note, font=font(18, "ui")), bottom + 20), note, fill=muted, font=font(18, "ui"))
    sheet.save(destination, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Render just this catalog component ID")
    parser.add_argument("--sheets-only", action="store_true")
    args = parser.parse_args()
    parts = json.loads((HARDWARE / "cad/catalog.json").read_text())["parts"]
    if args.only and args.only not in {part["id"] for part in parts}:
        parser.error(f"Unknown component: {args.only}")
    if not args.sheets_only:
        for part in parts:
            if args.only and part["id"] != args.only:
                continue
            render(read_stl(HARDWARE / part["file"]), HARDWARE / part["render"])
            print(f'Rendered {part["name"]}', flush=True)
    if all((HARDWARE / part["render"]).exists() for part in parts):
        exemplars = ["pedal-case-brake-u", "chestplate-mount", "pvc-cam-holder-left",
                     "imu-holster", "starlink-mount", "front-connector"]
        by_id = {part["id"]: part for part in parts}
        make_sheet([by_id[key] for key in exemplars], HARDWARE / "renders/overview.png", 3,
                   "Cart hardware", "Selected mechanical components from the cart.", cell_width=560)
        make_sheet(parts, HARDWARE / "renders/catalog.png", 7,
                   "Component catalog", "Mounts, mechanisms, and enclosures for the self-driving golf cart.")


if __name__ == "__main__":
    main()
