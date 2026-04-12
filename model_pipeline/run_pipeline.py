# model_pipeline/convert_to_onnx.py
"""Convert HuggingFace model to ONNX format using optimum exporters."""

import os
import logging
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export

logger = logging.getLogger(__name__)


def convert_model(model_name: str, output_dir: str, opset: int = 14) -> str:
    """
    Convert HuggingFace model to ONNX using optimum's modern export API.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save ONNX model
        opset: ONNX opset version (default: 14)
    
    Returns:
        Path to exported ONNX model file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {model_name}")
    
    # ✅ MODERN API: Use main_export instead of from_pretrained + opset
    try:
        main_export(
            model_name_or_path=model_name,
            output=output_path,
            task="sequence-classification",  # Explicit task for classification models
            opset=opset,
            device="cpu",  # CPU-only export (matches your GKE setup)
            no_post_process=False,  # Run graph optimizations during export
            framework="pt",  # PyTorch source
            trust_remote_code=False,  # Security best practice
        )
        logger.info(f"✅ Model exported to {output_path}")
        
    except Exception as e:
        # Fallback: manual export if main_export fails (rare)
        logger.warning(f"main_export failed ({e}), trying manual export...")
        return _manual_export_fallback(model_name, str(output_path), opset)
    
    # Verify output
    onnx_file = output_path / "model.onnx"
    if not onnx_file.exists():
        # optimum may save as <model_name>.onnx
        onnx_files = list(output_path.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]
            logger.info(f"Found ONNX model: {onnx_file.name}")
        else:
            raise FileNotFoundError(f"No ONNX file found in {output_path}")
    
    return str(onnx_file)


def _manual_export_fallback(model_name: str, output_dir: str, opset: int) -> str:
    """Fallback export using ORTModel (for edge cases)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Save tokenizer alongside model
    tokenizer.save_pretrained(output_dir)
    
    # Dummy input for tracing
    dummy_input = tokenizer(
        "Test input for ONNX export",
        return_tensors="pt",
        padding="max_length",
        max_length=128
    )
    
    # Export via torch.onnx
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"}
        }
    )
    
    logger.info(f"✅ Fallback export successful: {onnx_path}")
    return onnx_path