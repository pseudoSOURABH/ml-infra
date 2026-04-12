#!/usr/bin/env python3
"""Apply CPU-optimized INT8 dynamic quantization to ONNX model"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

INPUT_MODEL = Path("triton_model_repo/clinical_assertion/1/model.onnx")
OUTPUT_MODEL = Path("triton_model_repo/clinical_assertion/1/model_quantized.onnx")

def quantize():
    print("🔄 Applying INT8 dynamic quantization (CPU-optimized)...")
    
    # Dynamic quantization: best for CPU, no calibration data needed
    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QUInt8,  # Unsigned INT8 for CPU
        per_channel=True,
        reduce_range=True,  # Prevents overflow on CPU
        optimize_model=True
    )
    
    # Replace original with quantized
    INPUT_MODEL.unlink()
    OUTPUT_MODEL.rename(INPUT_MODEL)
    print(f"✅ Quantized model saved to: {INPUT_MODEL}")

if __name__ == "__main__":
    quantize()