"""
train.py
========
Trains the PVC classifier and saves it to models/pvc_cnn.keras

Usage:
    python src/train.py --mode demo          # smoke test, synthetic data
    python src/train.py --mode real          # real MIT-BIH (needs data/mitdb/*)
    python src/train.py --mode real --records 100 101 106 119 200 208 221
"""

import argparse
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_dataset_demo, load_dataset_real
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "real"], default="demo")
    parser.add_argument("--records", nargs="*", default=None,
                         help="MIT-BIH record names, e.g. 100 101 106")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", default="models/pvc_cnn.keras")
    parser.add_argument("--metrics_out", default="models/train_metrics.json",
                         help="Where to save training metrics as JSON, for "
                              "the app.py performance dashboard.")
    args = parser.parse_args()

    print(f"Loading data (mode={args.mode})...")
    if args.mode == "demo":
        X, y = load_dataset_demo(n_per_class=600)
    else:
        X, y = load_dataset_real(record_names=args.records)

    print(f"Dataset: {X.shape[0]} beats | "
          f"{int(y.sum())} PVC, {int((y == 0).sum())} normal")

    # normalize each beat independently (zero mean, unit max-abs)
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (np.max(np.abs(X), axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis]  # (N, L, 1) for Conv1D

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model(input_len=X.shape[1])
    model.summary()

    class_weight = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True)
        ],
        verbose=2,
    )

    print("\n=== Test set evaluation ===")
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    report_text = classification_report(y_test, y_pred, target_names=["Normal", "PVC"])
    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "PVC"],
                                         output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(report_text)
    print("Confusion matrix:\n", cm)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"\nSaved model to {args.out}")

    # Save metrics as JSON so app.py's performance dashboard can read real
    # numbers without needing to re-run training or parse console output.
    metrics = {
        "mode": args.mode,
        "records": args.records,
        "dataset": {
            "total_beats": int(X.shape[0]),
            "pvc_count": int(y.sum()),
            "normal_count": int((y == 0).sum()),
        },
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Normal", "PVC"],
    }
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()