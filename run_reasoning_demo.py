"""
run_reasoning_demo.py
=======================
Runs the reasoning-showcase beats (from generate_reasoning_demo_singles.py)
through the pipeline using the KNOWN-CORRECT R-peak position (index 90),
instead of predict_pipeline.py's general-purpose argmax(|signal|) estimate.

Why this is necessary: for a genuinely UNKNOWN uploaded image, the system
has no choice but to estimate the R-peak position (predict_pipeline.py
does this via argmax of the signal's absolute value). That estimate can
be wrong for beats with a very wide, bizarre QRS complex where the
S-trough is deeper than the R-peak is tall -- the argmax then lands on
the S-trough instead of the true R-peak, shifting every downstream
P-wave/QRS/T-wave search window and changing which reasoning criteria
fire (verified: a 53ms shift was measured on a synthetic wide-QRS test
beat). This is a genuine, real limitation of the general heuristic.

For THESE specific demo images, we built them ourselves directly from
real MIT-BIH beats, where the R-peak is always at index 90 by definition
(WINDOW_BEFORE, from data_loader.py's MIT-BIH annotation convention) --
so we can and should use that known-correct value directly here, rather
than re-deriving an uncertain estimate.

Run from the project root:
    py -3.12 run_reasoning_demo.py
"""

import sys
sys.path.insert(0, "src")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import glob
import numpy as np
import tensorflow as tf

from image_to_signal import digitize_ecg_image
from xai_explain import explain_prediction
from ecg_image_gen import signal_to_ecg_image

MODEL_PATH = "models/pvc_cnn.keras"
DEMO_DIR = "data/ecg_images/reasoning_demo"
OUT_DIR = "outputs/reasoning_demo"
KNOWN_R_PEAK = 90  # MIT-BIH WINDOW_BEFORE convention

os.makedirs(OUT_DIR, exist_ok=True)
model = tf.keras.models.load_model(MODEL_PATH)

image_paths = sorted(glob.glob(os.path.join(DEMO_DIR, "*.png")))
if not image_paths:
    print(f"No images found in {DEMO_DIR}. Run generate_reasoning_demo_singles.py first.")
    sys.exit(1)

for image_path in image_paths:
    name = os.path.splitext(os.path.basename(image_path))[0]

    signal, _, _, _, _ = digitize_ecg_image(image_path, out_len=200)
    signal_norm = signal - signal.mean()
    signal_norm = signal_norm / (np.max(np.abs(signal_norm)) + 1e-8)

    report = explain_prediction(model, signal_norm, r_peak_idx=KNOWN_R_PEAK)

    print(f"=== {name} ===")
    print(f"Prediction: {report['prediction']}  (p={report['pvc_probability']:.3f})")
    for r in report["reasons"]:
        print(f"  - {r}")

    hs, he = report["highlight_window"]
    is_pvc = report["prediction"] == "PVC"
    out_path = os.path.join(OUT_DIR, f"{name}_explained.png")
    title = f"Prediction: {report['prediction']} (p={report['pvc_probability']:.2f})"
    signal_to_ecg_image(signal, out_path, highlight_range=(hs, he) if is_pvc else None,
                         title=title)
    print(f"  Saved: {out_path}\n")

print("Done.")