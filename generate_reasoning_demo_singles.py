"""
generate_reasoning_demo_singles.py
====================================
Generates SINGLE-BEAT images (not multi-beat strips) for the specific
PVC beats verified to trigger ONLY P-wave-absence or ONLY QRS-widening
reasoning.

Why single-beat, not multi-beat: the multi-beat pipeline re-digitizes and
re-detects each R-peak from the rendered image (since it doesn't know
where beats are ahead of time), which can shift the effective R-peak
position by a few samples versus the original MIT-BIH annotation. For
beats that are boundary cases -- triggering exactly one reasoning
criterion -- that small shift can be enough to also trigger a second
criterion (e.g. T-wave) that wasn't present in the original scan.
Single-beat images digitize directly at the model's expected 200-sample
length with no separate peak-redetection step, so they stay faithful to
the beat exactly as it was when originally scanned.

Uses the exact indices found by your earlier category scan:
    p_wave_only examples: 8490, 4888, 4892, 4894, 4920
    qrs_only examples:    4199, 4202, 4206, 4208, 4211

Run from the project root:
    py -3.12 generate_reasoning_demo_singles.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from data_loader import load_dataset_real
from ecg_image_gen import signal_to_ecg_image

RECORDS = ['100', '101', '106', '119', '200', '201', '203', '208', '210',
           '213', '215', '219', '221', '223', '233']

print("Loading real MIT-BIH data...")
X, y = load_dataset_real(record_names=RECORDS)
print(f"Loaded {len(X)} beats\n")

P_WAVE_ONLY = [8490, 4888, 4892, 4894, 4920]
QRS_ONLY = [4199, 4202, 4206, 4208, 4211]

import os
os.makedirs("data/ecg_images/reasoning_demo", exist_ok=True)

print("Building P-wave-absence-only demo beats...")
for i, idx in enumerate(P_WAVE_ONLY):
    path = f"data/ecg_images/reasoning_demo/pwave_only_{i}.png"
    signal_to_ecg_image(X[idx], path)
    print(f"  {path}  (source index {idx})")

print("\nBuilding QRS-widened-only demo beats...")
for i, idx in enumerate(QRS_ONLY):
    path = f"data/ecg_images/reasoning_demo/qrs_only_{i}.png"
    signal_to_ecg_image(X[idx], path)
    print(f"  {path}  (source index {idx})")

print("\nDone. Run each through the SINGLE-BEAT pipeline, e.g.:")
print("  py -3.12 src/predict_pipeline.py --image data/ecg_images/reasoning_demo/pwave_only_0.png --model models/pvc_cnn.keras")
print("  py -3.12 src/predict_pipeline.py --image data/ecg_images/reasoning_demo/qrs_only_0.png --model models/pvc_cnn.keras")
print("\nIf a particular index doesn't show exactly the expected single reason,")
print("try the next one in the same category list -- these are boundary cases")
print("by definition, so a couple may drift with any reprocessing.")