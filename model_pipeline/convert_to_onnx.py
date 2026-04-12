#!/usr/bin/env python3
"""Convert Hugging Face model to ONNX format."""

import logging
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model(
    model_name: str,
    output_dir: str,
    opset: int = 13  # kept for CLI compatibility, no longer passed to from_pretrained
) -> Path:
    """
    Convert Hugging Face model to ONNX using optimum export API.

    Args:
        model_name: Hugging Face model identifier
        output_dir: Output directory for ONNX model and tokenizer
        opset: Retained as parameter for interface compatibility (not used internally)

    Returns:
        Path to converted ONNX model.onnx
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_name}")

    # FIX: Removed deprecated `from_transformers=True` and invalid `opset=` kwarg.
    # In optimum >= 1.14, use `export=True` only.
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    onnx_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    onnx_file = output_path / "model.onnx"
    if not onnx_file.exists():
        raise FileNotFoundError(
            f"Export succeeded but model.onnx not found in {output_path}. "
            f"Files present: {list(output_path.iterdir())}"
        )

    logger.info(f"Model saved to: {onnx_file}")
    return onnx_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bvanaken/clinical-assertion-negation-bert")
    parser.add_argument("--output", default="./model_repository/clinical_assertion/1")
    parser.add_argument("--opset", type=int, default=13)  # kept for CLI compat
    args = parser.parse_args()

    convert_model(args.model, args.output, args.opset)