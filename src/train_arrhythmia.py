"""
train_arrhythmia.py
====================
Trains the 4-class AAMI heartbeat CNN (N, S, V, F) on the inter-patient split
built by aami_dataset.py (train = DS1 minus 4 validation records, test = DS2).

Two variants (same file):
    signal only :  python src/train_arrhythmia.py --data aami_dataset.npz
    signal + R-R:  python src/train_arrhythmia.py --data aami_dataset.npz --use_rr \
                       --out models/arrhythmia_rr.keras \
                       --metrics models/arrhythmia_rr_metrics.json \
                       --confusion_png models/arrhythmia_rr_confusion.png

The architecture matches model.py (3 conv blocks, layer named 'last_conv' for
Grad-CAM) with a 4-way softmax head. With --use_rr a second input of five
R-R interval features (previous, next, local mean, and two ratios) is fused
with the CNN embedding: S and V beats are defined by how EARLY they arrive,
information the 200-sample window alone does not carry.

Per-beat normalisation is the same as train.py / predict_pipeline.py.
Class imbalance: softened class weights (inverse frequency ** p, default 0.5).
Model selection: mean validation F1 over classes that have >= 100 validation
beats (N, S, V). F has only a few dozen validation beats, which makes the
all-class macro-F1 too noisy to pick an epoch with.

Outputs (default names):
    models/arrhythmia_cnn.keras
    models/arrhythmia_metrics.json
    models/arrhythmia_confusion.png
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from aami_dataset import load_saved, CLASS_NAMES, SAMPLING_RATE

N_CLASSES = len(CLASS_NAMES)
PVC_INDEX = CLASS_NAMES.index("V")
N_RR = 5


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_arrhythmia_model(input_len=200, n_classes=N_CLASSES,
                           n_filters=(32, 64, 64), use_rr=False):
    sig_in = layers.Input(shape=(input_len, 1), name="ecg_signal")

    x = layers.Conv1D(n_filters[0], 7, padding="same", activation="relu")(sig_in)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(n_filters[1], 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(n_filters[2], 3, padding="same", activation="relu",
                      name="last_conv")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.GlobalAveragePooling1D()(x)

    inputs = [sig_in]
    if use_rr:
        rr_in = layers.Input(shape=(N_RR,), name="rr_features")
        r = layers.Dense(16, activation="relu")(rr_in)
        x = layers.Concatenate()([x, r])
        inputs.append(rr_in)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           name="class_probs")(x)

    model = models.Model(inputs, outputs,
                         name="arrhythmia_1dcnn_rr" if use_rr else "arrhythmia_1dcnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------
def normalize_np(X):
    """Same per-beat normalisation as train.py / predict_pipeline.py."""
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (np.max(np.abs(X), axis=1, keepdims=True) + 1e-8)
    return X[..., np.newaxis].astype("float32")


def rr_features(rr_prev, rr_next, rr_local, fs=SAMPLING_RATE):
    """Five R-R features (seconds / ratios), clipped to [0, 3]."""
    local = np.maximum(rr_local, 1.0)
    f = np.stack([rr_prev / fs, rr_next / fs, rr_local / fs,
                  rr_prev / local, rr_next / local], axis=1)
    return np.clip(f, 0.0, 3.0).astype("float32")


def _normalize_tf(x):
    x = x - tf.reduce_mean(x)
    return x / (tf.reduce_max(tf.abs(x)) + 1e-8)


def _augment_beat(x):
    """Random scale, baseline wander, noise and a small time shift (raw units)."""
    n = x.shape[0]
    amp = tf.reduce_max(tf.abs(x - tf.reduce_mean(x))) + 1e-6
    t = tf.linspace(0.0, 1.0, n)
    scale = tf.random.uniform([], 0.8, 1.2)
    freq = tf.random.uniform([], 0.5, 2.0)
    phase = tf.random.uniform([], 0.0, 6.2831853)
    wander = (tf.random.uniform([], 0.0, 0.15) * amp
              * tf.sin(6.2831853 * freq * t + phase))
    noise = (tf.random.normal([n]) * tf.random.uniform([], 0.0, 0.03) * amp)
    x = x * scale + wander + noise
    shift = tf.random.uniform([], -6, 7, dtype=tf.int32)
    return tf.roll(x, shift, axis=0)


def make_train_ds(X_raw, y, class_w, batch, augment, seed, rr=None):
    wt = tf.constant([class_w[i] for i in range(N_CLASSES)], dtype=tf.float32)
    if rr is None:
        ds = tf.data.Dataset.from_tensor_slices((X_raw.astype("float32"),
                                                 y.astype("int32")))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X_raw.astype("float32"), rr,
                                                 y.astype("int32")))
    ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)

    def prep(*args):
        x, yy = args[0], args[-1]
        if augment:
            x = _augment_beat(x)
        x = _normalize_tf(x)[..., tf.newaxis]
        inp = {"ecg_signal": x}
        if rr is not None:
            inp["rr_features"] = args[1]
        return inp, yy, tf.gather(wt, yy)

    return (ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch).prefetch(tf.data.AUTOTUNE))


def make_inputs(Xn, rr=None):
    inp = {"ecg_signal": Xn}
    if rr is not None:
        inp["rr_features"] = rr
    return inp


def softened_class_weights(y_train, power):
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = (len(y_train) / (N_CLASSES * counts)) ** power
    w = w / w[y_train].mean()          # average sample weight stays ~1
    return {i: float(w[i]) for i in range(N_CLASSES)}


# --------------------------------------------------------------------------
# callbacks / evaluation
# --------------------------------------------------------------------------
class ValScores(tf.keras.callbacks.Callback):
    """
    Adds to the logs each epoch:
        val_macro_f1 : mean F1 over all 4 classes
        val_score    : mean F1 over classes with >= min_support val beats
                       (the metric used for early stopping / best weights)
    """

    def __init__(self, Xv, yv, min_support=100):
        super().__init__()
        self.Xv, self.yv = Xv, yv
        support = np.bincount(yv, minlength=N_CLASSES)
        self.sel = [i for i in range(N_CLASSES) if support[i] >= min_support]
        if len(self.sel) < 2:               # tiny validation set: use all classes
            self.sel = [i for i in range(N_CLASSES) if support[i] > 0]
        print("model selection uses classes:", [CLASS_NAMES[i] for i in self.sel])

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xv, batch_size=1024, verbose=0).argmax(1)
        f_all = f1_score(self.yv, p, average=None,
                         labels=list(range(N_CLASSES)), zero_division=0)
        logs["val_macro_f1"] = float(f_all.mean())
        logs["val_score"] = float(f_all[self.sel].mean())


def evaluate(model, Xin, y, records):
    prob = model.predict(Xin, batch_size=1024, verbose=0)
    pred = prob.argmax(1)
    labels = list(range(N_CLASSES))
    rep = classification_report(y, pred, labels=labels,
                                target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    v_tp = cm[PVC_INDEX, PVC_INDEX]
    out = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels,
                                   zero_division=0)),
        "report": rep,
        "confusion_matrix": cm.tolist(),
        "pvc_sensitivity": float(v_tp / max(cm[PVC_INDEX].sum(), 1)),
        "pvc_ppv": float(v_tp / max(cm[:, PVC_INDEX].sum(), 1)),
    }
    dom = {}
    for i, name in enumerate(CLASS_NAMES):
        recs = records[y == i]
        if len(recs):
            u, c = np.unique(recs, return_counts=True)
            dom[name] = {"top_record": str(u[c.argmax()]),
                         "share": float(c.max() / c.sum())}
    out["largest_record_share_per_class"] = dom
    return out, cm, pred


def save_confusion_png(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=140)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%", ha="center",
                    va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="aami_dataset.npz")
    ap.add_argument("--out", default="models/arrhythmia_cnn.keras")
    ap.add_argument("--metrics", default="models/arrhythmia_metrics.json")
    ap.add_argument("--confusion_png", default="models/arrhythmia_confusion.png")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--weight_power", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--use_rr", action="store_true",
                    help="add the 5 R-R interval features as a second input")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label",
                    default="REAL MIT-BIH, inter-patient (train DS1 / test DS2)",
                    help="how to tag these metrics (real vs synthetic)")
    args = ap.parse_args()

    print("GPU devices:", tf.config.list_physical_devices("GPU") or "NONE (CPU run)")
    tf.keras.utils.set_random_seed(args.seed)
    data, splits = load_saved(args.data)
    X, y, rec = data["X"], data["y"], data["record"]
    tr, va, te = splits["train"], splits["val"], splits["test"]
    RR = (rr_features(data["rr_prev"], data["rr_next"], data["rr_local"])
          if args.use_rr else None)

    for name, m in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name:<6}{int(m.sum()):>7}  " + "  ".join(
            f"{c}={n}" for c, n in zip(CLASS_NAMES, np.bincount(y[m], minlength=N_CLASSES))))

    class_w = softened_class_weights(y[tr], args.weight_power)
    print("class weights:", {CLASS_NAMES[k]: round(v, 2) for k, v in class_w.items()})
    print("variant:", "signal + R-R features" if args.use_rr else "signal only")

    train_ds = make_train_ds(X[tr], y[tr], class_w, args.batch,
                             augment=not args.no_augment, seed=args.seed,
                             rr=None if RR is None else RR[tr])
    Xv = make_inputs(normalize_np(X[va]), None if RR is None else RR[va])
    Xt = make_inputs(normalize_np(X[te]), None if RR is None else RR[te])
    yv, yt = y[va], y[te]
    val_ds = tf.data.Dataset.from_tensor_slices((Xv, yv)).batch(1024)

    model = build_arrhythmia_model(input_len=X.shape[1], use_rr=args.use_rr)
    callbacks = [
        ValScores(Xv, yv),                       # must come first
        tf.keras.callbacks.EarlyStopping(
            monitor="val_score", mode="max", patience=args.patience,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_score", mode="max", factor=0.5, patience=3,
            min_lr=1e-5),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=callbacks, verbose=2)
    scores = hist.history["val_score"]
    best_epoch = int(np.argmax(scores)) + 1
    print(f"\nbest epoch {best_epoch}/{len(scores)}  (val_score {max(scores):.3f})")

    print("\n=== TEST (DS2, unseen patients) ===")
    res, cm, pred = evaluate(model, Xt, yt, rec[te])
    print(classification_report(yt, pred, labels=list(range(N_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):\n", cm)
    print(f"macro-F1 {res['macro_f1']:.3f} | accuracy {res['accuracy']:.3f} | "
          f"PVC sensitivity {res['pvc_sensitivity']:.3f} | PVC PPV {res['pvc_ppv']:.3f}")
    for c, d in res["largest_record_share_per_class"].items():
        if d["share"] > 0.6:
            print(f"  WARNING: {int(d['share']*100)}% of test class {c} beats "
                  f"come from record {d['top_record']} - read that class's score with care.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    metrics = {
        "data_source": args.label,
        "variant": "signal+rr" if args.use_rr else "signal",
        "split_counts": {n: np.bincount(y[m], minlength=N_CLASSES).tolist()
                         for n, m in [("train", tr), ("val", va), ("test", te)]},
        "class_names": CLASS_NAMES,
        "class_weights": {CLASS_NAMES[k]: v for k, v in class_w.items()},
        "epochs_run": len(scores),
        "best_epoch": best_epoch,
        "best_val_score": float(max(scores)),
        "augmentation": not args.no_augment,
        "test": res,
    }
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_png(cm, args.confusion_png,
                       f"Test confusion matrix ({metrics['variant']})\n{args.label}")
    print(f"\nSaved {args.out}, {args.metrics}, {args.confusion_png}")


if __name__ == "__main__":
    main()
