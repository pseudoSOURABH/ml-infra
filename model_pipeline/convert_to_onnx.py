#!/usr/bin/env python3
"""Convert HuggingFace model to ONNX format — reads from local cache only."""

import logging
import os
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The pipeline image bakes the model here at Docker build time.
# This path is the single source of truth — never changed at runtime.
_BAKED_MODEL_DIR = "/hf_cache/clinical_assertion_onnx"


def convert_model(model_name: str, output_dir: str) -> Path:
    """
    Copy the pre-baked ONNX model and tokenizer to output_dir.

    model_name is kept as a parameter for interface compatibility but is only
    used as a fallback label in logs — the actual files come from the baked
    cache inside the Docker image. TRANSFORMERS_OFFLINE=1 is set in the image
    so any accidental HF call will raise an error rather than silently download.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baked = Path(_BAKED_MODEL_DIR)
    if not baked.exists():
        raise RuntimeError(
            f"Baked model cache not found at {_BAKED_MODEL_DIR}. "
            "Rebuild the pipeline Docker image — the model is baked in at "
            "image build time via Dockerfile.pipeline."
        )

    logger.info(f"Loading model from baked cache: {_BAKED_MODEL_DIR}")

    # Load from local directory — no network call, TRANSFORMERS_OFFLINE=1 enforces this
    model = ORTModelForSequenceClassification.from_pretrained(
        str(baked),
        export=False,   # Already ONNX — never re-export
    )
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(str(baked))
    tokenizer.save_pretrained(str(output_path))

    onnx_file = output_path / "model.onnx"
    if not onnx_file.exists():
        raise FileNotFoundError(
            f"model.onnx missing in {output_path} after copy from baked cache. "
            f"Files present: {list(output_path.iterdir())}"
        )

    logger.info(f"ONNX model ready at: {onnx_file}")
    return onnx_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bvanaken/clinical-assertion-negation-bert")
    parser.add_argument("--output", default="./output/temp_conversion")
    args = parser.parse_args()
    convert_model(args.model, args.output)