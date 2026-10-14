#!/usr/bin/env python3
"""Export a HuggingFace SegFormer cityscapes checkpoint to ONNX and build a
TensorRT engine for it.

Default target: SegFormer-b5 cityscapes at 1024x1024, fp16 engine.

Usage:
    python onboard/export_segformer_trt.py
    python onboard/export_segformer_trt.py --variant b2 --size 1024
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


VARIANTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "models"


class SegFormerLogits(torch.nn.Module):
    """Thin wrapper so the ONNX graph exports `pixel_values -> logits` directly."""

    def __init__(self, hf_model: SegformerForSemanticSegmentation):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_onnx(variant: str, size: int, onnx_path: Path, opset: int) -> None:
    name = VARIANTS[variant]
    print(f"[export] loading {name}")
    hf = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    wrapper = SegFormerLogits(hf).eval()
    dummy = torch.zeros(1, 3, size, size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] tracing -> {onnx_path}")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool,
                 workspace_mib: int) -> None:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        raise SystemExit(f"trtexec not found at {trtexec}")
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
    ]
    if fp16:
        cmd.append("--fp16")
    print("[trt] " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"trtexec failed (code {proc.returncode})")
    print(f"[trt] wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="b5", choices=tuple(VARIANTS))
    p.add_argument("--size", type=int, default=1024,
                   help="Square input resolution baked into the engine.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--workspace-mib", type=int, default=4096)
    p.add_argument("--fp32", action="store_true",
                   help="Build an fp32 engine instead of fp16.")
    p.add_argument("--skip-onnx", action="store_true",
                   help="Skip ONNX export (engine build only).")
    p.add_argument("--skip-engine", action="store_true",
                   help="Skip engine build (ONNX export only).")
    args = p.parse_args()

    stem = f"segformer_{args.variant}_cityscapes_{args.size}"
    onnx_path = args.out_dir / f"{stem}.onnx"
    suffix = "fp32" if args.fp32 else "fp16"
    engine_path = args.out_dir / f"{stem}_{suffix}.engine"

    if not args.skip_onnx:
        export_onnx(args.variant, args.size, onnx_path, args.opset)
    if not args.skip_engine:
        build_engine(onnx_path, engine_path, fp16=not args.fp32,
                     workspace_mib=args.workspace_mib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
