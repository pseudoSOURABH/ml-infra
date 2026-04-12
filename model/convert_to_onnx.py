#!/usr/bin/env python3
"""Convert Hugging Face model to ONNX format for Triton CPU inference"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

MODEL_NAME = "bvanaken/clinical-assertion-negation-bert"
OUTPUT_DIR = Path("triton_model_repo/clinical_assertion/1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_onnx():
    print(f"🔄 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Dummy input for tracing
    dummy_input = tokenizer(
        "The patient denies chest pain.",
        return_tensors="pt",
        padding="max_length",
        max_length=128
    )
    
    onnx_path = OUTPUT_DIR / "model.onnx"
    
    print("🔄 Converting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"✅ ONNX model saved to: {onnx_path}")
    return tokenizer, model

if __name__ == "__main__":
    convert_to_onnx()