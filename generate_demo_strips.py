"""
generate_reasoning_demo_strips.py
===================================
Generates multi-beat strips using SPECIFIC real PVC beats already verified
(from your earlier clinical_reasoning() scan) to trigger particular XAI
reasoning types -- useful for demoing that the explanation genuinely
varies per beat, not just always saying the same thing.

Uses the exact indices found by your category scan:
    p_wave_only examples: 8490, 4888, 4892, 4894, 4920
    qrs_only examples:    4199, 4202, 4206, 4208, 4211

Run from the project root:
    py -3.12 generate_reasoning_demo_strips.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from data_loader import load_dataset_real
from ecg_image_gen import build_multi_beat_strip

RECORDS = ['100', '101', '106', '119', '200', '201', '203', '208', '210',
           '213', '215', '219', '221', '223', '233']

print("Loading real MIT-BIH data...")
X, y = load_dataset_real(record_names=RECORDS)
idx_norm = np.where(y == 0)[0]
print(f"Loaded {len(X)} beats\n")

# Verified indices from your earlier category scan -- each of these PVC
# beats is confirmed to trigger ONLY the named reasoning type (not a mix).
P_WAVE_ONLY = [8490, 4888, 4892, 4894, 4920]
QRS_ONLY = [4199, 4202, 4206, 4208, 4211]


def build(name, beat_specs):
    """beat_specs: list of (kind, value).
    kind='norm' -> value is an offset into idx_norm
    kind='pvc'  -> value is a DIRECT index into X (from the category lists above)"""
    seq, labels = [], []
    for kind, value in beat_specs:
        if kind == "pvc":
            seq.append(X[value])
            labels.append("PVC")
        else:
            seq.append(X[idx_norm[value]])
            labels.append("norm")
    out_path = f"data/ecg_images/{name}.png"
    build_multi_beat_strip(seq, out_path)
    print(f"{name}.png  ->  {len(seq)} beats, true composition: {labels}")
    return out_path


# 1) P-wave-absence-only PVCs -- explanation should show ONLY the
#    "No discernible P wave detected..." reason, no QRS/T-wave mention.
build("demo_strip_pwave_absent", [
    ("norm", 90), ("pvc", P_WAVE_ONLY[0]), ("norm", 91),
    ("norm", 92), ("pvc", P_WAVE_ONLY[1]), ("norm", 93),
])

# 2) QRS-widened-only PVCs -- explanation should show ONLY the
#    "QRS complex is widened..." reason, no P-wave/T-wave mention.
build("demo_strip_qrs_widened", [
    ("norm", 94), ("pvc", QRS_ONLY[0]), ("norm", 95),
    ("norm", 96), ("pvc", QRS_ONLY[1]), ("norm", 97),
])

# 3) Mixed strip: one of each type in a single strip -- good for showing
#    the explanation genuinely differs beat-to-beat, not templated.
build("demo_strip_mixed_reasoning", [
    ("norm", 98), ("pvc", P_WAVE_ONLY[2]), ("norm", 99),
    ("pvc", QRS_ONLY[2]), ("norm", 100), ("pvc", P_WAVE_ONLY[3]),
])

print("\nAll strips saved to data/ecg_images/")
print("Expected explanation content per strip:")
print("  demo_strip_pwave_absent.png   -> PVC beats show ONLY the P-wave-absent reason")
print("  demo_strip_qrs_widened.png    -> PVC beats show ONLY the QRS-widened reason")
print("  demo_strip_mixed_reasoning.png -> different reasons on different beats")
print("\nRun with:")
print("  py -3.12 src/multi_beat_pipeline.py --image data/ecg_images/demo_strip_pwave_absent.png --model models/pvc_cnn.keras")