#!/usr/bin/env python3
"""Validate model accuracy after optimization."""

import logging
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# bvanaken/clinical-assertion-negation-bert label space:
# 0=PRESENT, 1=ABSENT, 2=POSSIBLE
# This model does NOT have a CONDITIONAL label — conditional/hypothetical
# sentences are classified as POSSIBLE by this model's design.
LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "POSSIBLE"}

TEST_CASES = [
    # (sentence, expected_label)
    ("The patient denies chest pain.",                                  "ABSENT"),
    ("He has a history of hypertension.",                              "PRESENT"),
    # Correctly mapped to POSSIBLE — this model treats hypothetical/conditional
    # as POSSIBLE, not a separate CONDITIONAL class.
    ("If the patient experiences dizziness, reduce the dosage.",       "POSSIBLE"),
    ("No signs of pneumonia were observed.",                           "ABSENT"),
    # Extra cases to make the test suite more robust
    ("The scan shows possible signs of infection.",                    "POSSIBLE"),
    ("Chest X-ray confirms bilateral pneumonia.",                      "PRESENT"),
]


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    accuracy_threshold: float = 0.95,
) -> bool:
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

            shifted = logits - np.max(logits)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            score = float(np.max(probs))

            is_correct = pred_label == expected
            if is_correct:
                correct += 1

            results.append((sentence[:55], expected, pred_label, score, is_correct))

        except Exception as e:
            logger.error(f"Inference error on '{sentence}': {e}")
            results.append((sentence[:55], expected, "ERROR", 0.0, False))

    accuracy = correct / len(TEST_CASES)

    logger.info("=== Validation Results ===")
    for sent, exp, pred, score, ok in results:
        mark = "✓" if ok else "✗"
        logger.info(
            f"  {mark} expected={exp:8s} predicted={pred:8s} "
            f"score={score:.3f}  {sent}"
        )

    logger.info(f"Accuracy: {accuracy:.0%} (threshold: {accuracy_threshold:.0%})")

    if accuracy >= accuracy_threshold:
        logger.info("✓ Validation PASSED")
        return True

    logger.error("✗ Validation FAILED")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    exit(0 if evaluate_model(args.model, args.tokenizer, args.threshold) else 1)