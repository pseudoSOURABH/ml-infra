#!/usr/bin/env python3
"""
Probe bvanaken/clinical-assertion-negation-bert to see what it ACTUALLY
predicts, so we can write test cases that reflect real model behavior.
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "bvanaken/clinical-assertion-negation-bert"
LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "POSSIBLE"}

# Broad set of probes covering all three classes
PROBES = [
    # Clear PRESENT
    "He has a history of hypertension.",
    "Chest X-ray confirms bilateral pneumonia.",
    "The patient has diabetes.",
    "She was diagnosed with breast cancer.",
    # Clear ABSENT
    "The patient denies chest pain.",
    "No signs of pneumonia were observed.",
    "There is no evidence of fracture.",
    "Patient denies shortness of breath.",
    # Hypothetical / conditional — what does the model ACTUALLY say?
    "If the patient experiences dizziness, reduce the dosage.",
    "The scan shows possible signs of infection.",
    "Patient may have early-stage pneumonia.",
    "There is a possibility of cardiac involvement.",
    "The patient might have appendicitis.",
    "Possible fracture of the left tibia.",
    "Rule out pulmonary embolism.",
    "Cannot exclude malignancy.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

print(f"\n{'Sentence':<55} {'Pred':<10} {'Scores (P/A/Po)'}")
print("-" * 85)

for sentence in PROBES:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits.numpy()
    
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    pred = LABEL_MAP[int(np.argmax(probs))]
    p, a, po = probs[0]
    
    print(f"{sentence:<55} {pred:<10} P={p:.2f} A={a:.2f} Po={po:.2f}")