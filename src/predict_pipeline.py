"""
predict_pipeline.py
====================
THE FULL PIPELINE your project describes, wired end-to-end:

    ECG image
        --(Computer Vision: image_to_signal.py)-->  1D signal
        --(ML: model.py, trained CNN)-->             PVC / Normal + probability
        --(Explainable AI: xai_explain.py)-->         Grad-CAM highlight + clinical reasons
        --(ecg_image_gen.py)-->                       final image with PVC region highlighted

Usage:
    python src/predict_pipeline.py --image path/to/ecg.png --model models/pvc_cnn.keras
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import tensorflow as tf

from image_to_signal import digitize_ecg_image
from xai_explain import explain_prediction
from ecg_image_gen import signal_to_ecg_image


def run_pipeline(image_path, model_path, out_dir="outputs", sampling_rate=360):
    os.makedirs(out_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    beat_len = model.input_shape[1]

    # 1) Computer Vision: image -> raw digitized signal
    signal, col2idx, idx2col, mask, shape = digitize_ecg_image(
        image_path, out_len=beat_len)

    # 2) normalize the same way training data was normalized
    signal_norm = signal - signal.mean()
    signal_norm = signal_norm / (np.max(np.abs(signal_norm)) + 1e-8)

    # 3) ML + XAI: predict + explain (Grad-CAM gives highlight window)
    r_peak_idx = int(np.argmax(np.abs(signal_norm)))  # simple R-peak estimate
    report = explain_prediction(model, signal_norm, r_peak_idx=r_peak_idx,
                                 sampling_rate=sampling_rate)

    # 4) Render final output image using the ORIGINAL digitized signal.
    #    Only draw the highlight region when the model actually predicted
    #    PVC -- a Grad-CAM window is still computed either way internally,
    #    but showing a red "abnormality" band on a Normal beat is
    #    misleading, so we suppress it for Normal predictions.
    hs, he = report["highlight_window"]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_explained.png")

    title = (f"Prediction: {report['prediction']} "
             f"(p={report['pvc_probability']:.2f})")

    is_pvc = report["prediction"] == "PVC"
    signal_to_ecg_image(
        signal, out_path,
        highlight_range=(hs, he) if is_pvc else None,
        title=title,
    )

    result = {
        "prediction": report["prediction"],
        "pvc_probability": report["pvc_probability"],
        "highlight_window_samples": (hs, he),
        "reasons": report["reasons"],
        "output_image": out_path,
        "signal": signal,
        "saliency_curve": report["saliency_curve"],
    }
    return result


def print_report(result):
    print("=" * 60)
    print(f"PREDICTION: {result['prediction']}  "
          f"(PVC probability = {result['pvc_probability']:.3f})")
    print("-" * 60)
    print("Explanation:")
    for r in result["reasons"]:
        print(f"  - {r}")
    print(f"\nHighlighted output image: {result['output_image']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/pvc_cnn.keras")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    result = run_pipeline(args.image, args.model, args.out_dir)
    print_report(result)