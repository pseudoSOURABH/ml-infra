#!/usr/bin/env python3
"""Quick size comparison: HuggingFace model vs optimized pipeline output."""

from huggingface_hub import snapshot_download
from pathlib import Path
import shutil, tempfile

def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024

# ── Original HF model ────────────────────────────────────────────────────────
print("Downloading original model...")
hf_dir = Path(snapshot_download("bvanaken/clinical-assertion-negation-bert"))
hf_mb = dir_size_mb(hf_dir)

# ── Pipeline output ──────────────────────────────────────────────────────────
pipeline_output = Path("../local_test/pipeline_output_quant/model_repository")
onnx_file = pipeline_output / "clinical_assertion/1/model.onnx"

onnx_mb = onnx_file.stat().st_size / 1024 / 1024 if onnx_file.exists() else None
repo_mb  = dir_size_mb(pipeline_output) if pipeline_output.exists() else None

# ── Report ───────────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
print(f"  Original HF model (all files):  {hf_mb:>8.1f} MB")
if onnx_mb:
    print(f"  Quantized ONNX (model.onnx):    {onnx_mb:>8.1f} MB  ({100*(1-onnx_mb/hf_mb):.0f}% smaller)")
if repo_mb:
    print(f"  Full Triton repo (with tokens): {repo_mb:>8.1f} MB  ({100*(1-repo_mb/hf_mb):.0f}% smaller)")
print(f"{'='*45}\n")