#!/usr/bin/env python3
"""Apply CPU optimizations to ONNX model."""

import logging
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
    Optimize ONNX model for CPU inference.

    Args:
        model_path: Path to input ONNX model file (string or Path)
        output_path: Directory path for optimized output (string or Path)
        quantize: Apply INT8 dynamic quantization
        level: Optimization level (O1, O2, O3, O4)

    Returns:
        Path to optimized model.onnx
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model for optimization: {model_path}")

    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path.parent),
        file_name=model_path.name
    )

    # Map string level to integer — optimum 1.21 OptimizationConfig expects int 1-4
    level_map = {"O1": 1, "O2": 2, "O3": 3, "O4": 4}
    optimization_level_int = level_map.get(level, 2)

    # FIXED: Only these fields are valid in optimum 1.21.0 OptimizationConfig.
    # Removed: enable_layer_norm, enable_gemm_fast_gelu (both invalid in this version)
    optimization_config = OptimizationConfig(
        optimization_level=optimization_level_int,
        optimize_for_gpu=False,
        optimize_with_onnxruntime_only=True,
        enable_gelu_approximation=True,
        enable_transformers_specific_optimizations=True,
        fp16=False,
    )

    logger.info(f"Applying graph optimizations (level={level})")
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=str(output_path),
    )

    if quantize:
        logger.info("Applying INT8 dynamic quantization")
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        # avx2 is the correct choice for x86_64 Cloud Build machines (not arm64)
        qconfig = AutoQuantizationConfig.avx2(
            is_static=False,
            per_channel=False,
        )

        optimizer.quantize(
            quantization_config=qconfig,
            save_dir=str(output_path),
        )

        # optimum 1.21 names output: model_quantized.onnx
        candidates = list(output_path.glob("*quantized*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                f"Quantization completed but no *quantized*.onnx found in {output_path}. "
                f"Files present: {list(output_path.iterdir())}"
            )
        optimized_file = candidates[0]
    else:
        candidates = (
            list(output_path.glob("*optimized*.onnx"))
            or list(output_path.glob("model.onnx"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"Optimization completed but no output .onnx found in {output_path}. "
                f"Files present: {list(output_path.iterdir())}"
            )
        optimized_file = candidates[0]

    # Normalize filename to model.onnx for Triton
    final_path = output_path / "model.onnx"
    if optimized_file.resolve() != final_path.resolve():
        optimized_file.rename(final_path)

    logger.info(f"Optimized model saved to: {final_path}")
    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--level", default="O2", choices=["O1", "O2", "O3", "O4"])
    args = parser.parse_args()

    optimize_for_cpu(
        args.input,
        args.output,
        quantize=not args.no_quantize,
        level=args.level
    )