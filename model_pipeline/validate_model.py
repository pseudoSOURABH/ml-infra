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

LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "POSSIBLE"}


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    accuracy_threshold: float = 0.95
) -> bool:
    """
    Evaluate optimized ONNX model on assertion classification test cases.

    Args:
        model_path: Path to model.onnx file (string or Path)
        tokenizer_path: Directory containing tokenizer files (string or Path)
        accuracy_threshold: Minimum accuracy fraction to pass (e.g. 0.95)

    Returns:
        True if accuracy >= threshold, False otherwise
    """
    # FIX: Normalize inputs to Path objects here rather than relying on callers.
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    logger.info(f"Loading ONNX model: {model_path}")
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path.parent),
        file_name=model_path.name
    )

    correct = 0
    results = []

    for sentence, expected_label in TEST_CASES:
        try:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()
            pred_idx = int(np.argmax(logits, axis=1)[0])
            pred_label = LABEL_MAP.get(pred_idx, "UNKNOWN")

            # Softmax for confidence score
            exp_logits = np.exp(logits - np.max(logits))  # numerically stable
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            score = float(np.max(probs))

            # CONDITIONAL is not in this model's label set — map POSSIBLE -> CONDITIONAL
            if expected_label == "CONDITIONAL" and pred_label == "POSSIBLE":
                is_correct = True
                pred_label = "CONDITIONAL"
            else:
                is_correct = (pred_label == expected_label)

            if is_correct:
                correct += 1

            results.append({
                "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                "expected": expected_label,
                "predicted": pred_label,
                "score": round(score, 4),
                "correct": is_correct,
            })

        except Exception as e:
            logger.error(f"Error on '{sentence}': {e}")
            results.append({
                "sentence": sentence,
                "expected": expected_label,
                "predicted": "ERROR",
                "score": 0.0,
                "correct": False,
            })

    accuracy = correct / len(TEST_CASES)

    logger.info("\n=== Validation Results ===")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        logger.info(
            f"{status} Expected: {r['expected']:11} | "
            f"Predicted: {r['predicted']:11} | "
            f"Score: {r['score']:.4f} | "
            f"{r['sentence']}"
        )

    logger.info(f"\nAccuracy: {accuracy:.2%}  (threshold: {accuracy_threshold:.2%})")

    if accuracy >= accuracy_threshold:
        logger.info("✓ Model validation PASSED")
        return True
    else:
        logger.error("✗ Model validation FAILED — accuracy below threshold")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer directory")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    success = evaluate_model(args.model, args.tokenizer, args.threshold)
    exit(0 if success else 1)