#!/usr/bin/env python3
"""
Quantize ONNX model for CPU inference using onnxruntime directly.

Deliberately does NOT use optimum's ORTOptimizer or ORTQuantizer —
those APIs have broken across minor versions repeatedly.
onnxruntime.quantization is the stable low-level API.
"""

import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_for_cpu(
    model_path: str,
    output_path: str,
    quantize: bool = True,
    level: str = "O2",   # kept for interface compat, not used
) -> Path:
    """
    Quantize an ONNX model using onnxruntime.quantization.quantize_dynamic.

    This is intentionally simple — no optimum optimizer calls.
    quantize_dynamic is stable across onnxruntime versions and needs
    zero configuration beyond input/output paths and weight type.

    Args:
        model_path: Path to source model.onnx
        output_path: Output directory
        quantize: If False, just copies model as-is (for debugging)
        level: Ignored — kept for call-site compatibility with run_pipeline.py

    Returns:
        Path to final model.onnx
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    final_path = output_path / "model.onnx"

    if not quantize:
        logger.info("Quantization skipped — copying model as-is")
        shutil.copy2(str(model_path), str(final_path))
        return final_path

    logger.info(f"Applying INT8 dynamic quantization to: {model_path}")

    # onnxruntime.quantization is stable since onnxruntime 1.8 — no API changes
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(final_path),
        weight_type=QuantType.QInt8,
    )

    if not final_path.exists():
        raise FileNotFoundError(
            f"quantize_dynamic ran but output not found at {final_path}"
        )

    original_mb = model_path.stat().st_size / 1024 / 1024
    quantized_mb = final_path.stat().st_size / 1024 / 1024
    logger.info(
        f"Quantization complete: {original_mb:.1f}MB -> {quantized_mb:.1f}MB "
        f"({100*(1 - quantized_mb/original_mb):.0f}% reduction)"
    )
    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    optimize_for_cpu(args.input, args.output, quantize=not args.no_quantize)