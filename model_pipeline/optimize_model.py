#!/usr/bin/env python3
"""Apply CPU optimizations to ONNX model."""

import logging
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_for_cpu(
    model_path: str,
    output_path: str,
    quantize: bool = True,
    level: str = "O2"
) -> Path:
    """
    Optimize and optionally quantize an ONNX model for CPU inference.

    Uses ORTOptimizer for graph optimization and ORTQuantizer for INT8
    dynamic quantization. These are two separate classes in optimum 1.21 —
    ORTOptimizer does NOT have a .quantize() method.

    Args:
        model_path: Path to input model.onnx file
        output_path: Directory for output artifacts
        quantize: Whether to apply INT8 dynamic quantization after optimization
        level: Graph optimization level O1-O4 (mapped to int 1-4)

    Returns:
        Path to final model.onnx
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    opt_dir = output_path / "optimized"
    quant_dir = output_path / "quantized"

    opt_dir.mkdir(parents=True, exist_ok=True)

    # ── Step A: Graph optimization via ORTOptimizer ──────────────────────────
    level_map = {"O1": 1, "O2": 2, "O3": 3, "O4": 4}
    opt_level = level_map.get(level, 2)

    logger.info(f"Loading model for optimization: {model_path}")
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path.parent),
        file_name=model_path.name,
    )

    # Only use fields that are valid in optimum 1.21.0 OptimizationConfig.
    # enable_layer_norm, enable_gemm_fast_gelu are NOT valid — removed.
    optimization_config = OptimizationConfig(
        optimization_level=opt_level,
        optimize_for_gpu=False,
        optimize_with_onnxruntime_only=True,
        enable_gelu_approximation=True,
        fp16=False,
    )

    logger.info(f"Applying graph optimizations (level={level})")
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=str(opt_dir),
    )

    # Locate the optimized file — optimum 1.21 names it model_optimized.onnx
    opt_candidates = list(opt_dir.glob("*optimized*.onnx")) or list(opt_dir.glob("model.onnx"))
    if not opt_candidates:
        raise FileNotFoundError(
            f"Optimization produced no .onnx file in {opt_dir}. "
            f"Files present: {list(opt_dir.iterdir())}"
        )
    optimized_file = opt_candidates[0]
    logger.info(f"Optimized model: {optimized_file}")

    # ── Step B: INT8 quantization via ORTQuantizer (separate class!) ─────────
    # NOTE: ORTOptimizer does NOT have a .quantize() method in optimum 1.21.
    # Quantization requires ORTQuantizer loaded from the *optimized* model dir.
    if quantize:
        quant_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Applying INT8 dynamic quantization via ORTQuantizer")

        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(
            str(opt_dir),
            file_name=optimized_file.name,
        )

        # avx2: correct for x86_64 Cloud Build machines (NOT arm64)
        qconfig = AutoQuantizationConfig.avx2(
            is_static=False,   # dynamic quantization — safer for accuracy
            per_channel=False,
        )

        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=str(quant_dir),
        )

        # optimum 1.21 ORTQuantizer names output: model_quantized.onnx
        quant_candidates = list(quant_dir.glob("*quantized*.onnx")) or list(quant_dir.glob("model.onnx"))
        if not quant_candidates:
            raise FileNotFoundError(
                f"Quantization produced no .onnx file in {quant_dir}. "
                f"Files present: {list(quant_dir.iterdir())}"
            )
        final_source = quant_candidates[0]
        logger.info(f"Quantized model: {final_source}")
    else:
        final_source = optimized_file

    # ── Step C: Normalize to model.onnx for Triton ───────────────────────────
    final_path = output_path / "model.onnx"
    shutil.copy2(str(final_source), str(final_path))
    logger.info(f"Final model saved to: {final_path}")
    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input ONNX model file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--level", default="O2", choices=["O1", "O2", "O3", "O4"])
    args = parser.parse_args()

    optimize_for_cpu(
        args.input,
        args.output,
        quantize=not args.no_quantize,
        level=args.level
    )