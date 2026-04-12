#!/usr/bin/env python3
"""Apply CPU optimizations to ONNX model."""

import logging
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_for_cpu(
    model_path: Path,
    output_path: Path,
    quantize: bool = True,
    level: str = "O2"
) -> Path:
    """
    Optimize ONNX model for CPU inference.
    
    Args:
        model_path: Path to input ONNX model
        output_path: Path for optimized model
        quantize: Apply INT8 quantization
        level: Optimization level (O1, O2, O3, O4)
        
    Returns:
        Path to optimized model
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model for optimization: {model_path}")
    
    # Load model
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path.parent),
        file_name=model_path.name
    )
    
    # Create optimization config for CPU
    optimization_config = OptimizationConfig(
        optimization_level=level,
        optimize_for_gpu=False,
        optimize_with_onnxruntime_only=True,
        # Graph optimizations
        enable_gelu_approximation=True,
        enable_gemm_fast_gelu=True,
        enable_layer_norm=True,
        # Memory optimizations
        enable_transformers_specific_optimizations=True,
        fp16=False,  # CPU doesn't benefit from FP16 typically
    )
    
    logger.info(f"Applying optimizations (level={level})")
    optimizer = ORTOptimizer.from_pretrained(model)
    optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=output_path,
        file_suffix="optimized" if not quantize else None,
    )
    
    if quantize:
        logger.info("Applying INT8 dynamic quantization")
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False,  # Dynamic quantization (safer for accuracy)
            per_channel=False,
        )
        optimizer.quantize(
            quantization_config=qconfig,
            save_dir=output_path,
            file_suffix="quantized",
        )
        optimized_file = output_path / "model.quantized.onnx"
    else:
        optimized_file = output_path / "model.optimized.onnx"
    
    # Rename to standard name for Triton
    final_path = output_path / "model.onnx"
    if optimized_file.exists() and optimized_file != final_path:
        optimized_file.rename(final_path)
    
    logger.info(f"Optimized model saved to: {final_path}")
    return final_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--level", default="O2", choices=["O1", "O2", "O3", "O4"])
    args = parser.parse_args()
    
    optimize_for_cpu(
        Path(args.input),
        args.output,
        quantize=not args.no_quantize,
        level=args.level
    )