#!/usr/bin/env python3
"""Validate model with required test cases before deployment"""

import json
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

MODEL_PATH = Path("triton_model_repo/clinical_assertion/1/model.onnx")
TOKENIZER_NAME = "bvanaken/clinical-assertion-negation-bert"

# 🧪 Required test cases
TEST_CASES = [
    ("The patient denies chest pain.", "ABSENT"),
    ("He has a history of hypertension.", "PRESENT"),
    ("If the patient experiences dizziness, reduce the dosage.", "CONDITIONAL"),
    ("No signs of pneumonia were observed.", "ABSENT"),
]

LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "CONDITIONAL"}  # Verify with model config

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return tokenizer, session

def predict(tokenizer, session, sentence: str):
    inputs = tokenizer(
        sentence,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )
    logits = outputs[0][0]
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    pred_idx = probs.argmax()
    return LABEL_MAP.get(pred_idx, "UNKNOWN"), float(probs[pred_idx])

def run_tests():
    tokenizer, session = load_model()
    print("🧪 Running validation tests...\n")
    
    all_passed = True
    for sentence, expected in TEST_CASES:
        label, score = predict(tokenizer, session, sentence)
        status = "✅ PASS" if label == expected else "❌ FAIL"
        if label != expected:
            all_passed = False
        print(f"{status} | Input: {sentence[:50]}...")
        print(f"       Expected: {expected}, Got: {label} (score: {score:.4f})\n")
    
    if all_passed:
        print("🎉 All tests passed! Model ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Review before deploying.")
        return False

if __name__ == "__main__":
    import torch
    run_tests()