#!/usr/bin/env python3
"""Validate model accuracy after optimization."""

import logging
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_CASES = [
    ("The patient denies chest pain.", "ABSENT"),
    ("He has a history of hypertension.", "PRESENT"),
    ("If the patient experiences dizziness, reduce the dosage.", "CONDITIONAL"),
    ("No signs of pneumonia were observed.", "ABSENT"),
]

# bvanaken/clinical-assertion-negation-bert label mapping
LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "POSSIBLE"}


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    accuracy_threshold: float = 0.95,
) -> bool:
    """
    Run test cases against the ONNX model and check accuracy threshold.

    Args:
        model_path: Path to model.onnx
        tokenizer_path: Directory containing tokenizer files
        accuracy_threshold: Float 0-1, minimum pass accuracy

    Returns:
        True if accuracy >= threshold
    """
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    logger.info(f"Loading model from: {model_path}")
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path.parent),
        file_name=model_path.name,
    )

    correct = 0
    results = []

    for sentence, expected in TEST_CASES:
        try:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()
            pred_idx = int(np.argmax(logits, axis=1)[0])
            pred_label = LABEL_MAP.get(pred_idx, "UNKNOWN")

            # Numerically stable softmax for confidence
            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            score = float(np.max(probs))

            # CONDITIONAL maps to POSSIBLE in this model's label set
            is_correct = (pred_label == expected) or (
                expected == "CONDITIONAL" and pred_label == "POSSIBLE"
            )
            if is_correct:
                correct += 1
                pred_label = expected  # normalise for display

            results.append((sentence[:55], expected, pred_label, score, is_correct))

        except Exception as e:
            logger.error(f"Inference error on '{sentence}': {e}")
            results.append((sentence[:55], expected, "ERROR", 0.0, False))

    accuracy = correct / len(TEST_CASES)

    logger.info("=== Validation Results ===")
    for sent, exp, pred, score, ok in results:
        mark = "✓" if ok else "✗"
        logger.info(f"  {mark} expected={exp:11s} predicted={pred:11s} score={score:.3f}  {sent}")

    logger.info(f"Accuracy: {accuracy:.0%} (threshold: {accuracy_threshold:.0%})")

    if accuracy >= accuracy_threshold:
        logger.info("✓ Validation PASSED")
        return True

    logger.error("✗ Validation FAILED")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    exit(0 if evaluate_model(args.model, args.tokenizer, args.threshold) else 1)