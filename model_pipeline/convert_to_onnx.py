#!/usr/bin/env python3
"""Convert Hugging Face model to ONNX format."""

import os
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(
    model_name: str,
    output_dir: str,
    opset: int = 13
) -> Path:
    """
    Convert Hugging Face model to ONNX.
    
    Args:
        model_name: Hugging Face model identifier
        output_dir: Output directory for ONNX model
        opset: ONNX opset version
        
    Returns:
        Path to converted ONNX model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Export to ONNX using optimum
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        from_transformers=True,
        export=True,
        opset=opset
    )
    
    # Save model and tokenizer
    onnx_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Model saved to: {output_path}")
    return output_path / "model.onnx"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bvanaken/clinical-assertion-negation-bert")
    parser.add_argument("--output", default="./model_repository/clinical_assertion/1")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    
    convert_model(args.model, args.output, args.opset)