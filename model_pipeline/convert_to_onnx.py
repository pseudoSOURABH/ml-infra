#!/usr/bin/env python3
"""Convert HuggingFace model to ONNX format."""

import logging
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model(model_name: str, output_dir: str) -> Path:
    """
    Export a HuggingFace model to ONNX.
    Returns path to model.onnx.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_name}")

    # export=True is the only correct API in optimum >= 1.14
    # Never pass opset= or from_transformers= here
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
    )
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path))

    onnx_file = output_path / "model.onnx"
    if not onnx_file.exists():
        raise FileNotFoundError(
            f"Export completed but model.onnx missing in {output_path}. "
            f"Files: {list(output_path.iterdir())}"
        )

    logger.info(f"ONNX model saved to: {onnx_file}")
    return onnx_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bvanaken/clinical-assertion-negation-bert")
    parser.add_argument("--output", default="./output/temp_conversion")
    args = parser.parse_args()
    convert_model(args.model, args.output)