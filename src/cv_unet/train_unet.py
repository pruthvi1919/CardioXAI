"""
train_unet.py
==============
Trains the U-Net trace segmenter on freshly generated synthetic ECG images
(infinite stream, different every step) built from real MIT-BIH beats.

    python src/cv_unet/train_unet.py --bank aami_dataset.npz
    python src/cv_unet/train_unet.py --bank aami_dataset.npz --epochs 30 --steps 150 --batch 8

Beats for training images come from the TRAIN split (DS1 minus validation
records); validation images use the validation records, with fixed seeds.

IMPORTANT: every metric printed here is on SYNTHETIC validation images. It
shows the network learned the task; it is NOT an accuracy figure for real
photographs. Use eval_digitizers.py and your own real ECG images for that.

Outputs:
    models/unet_trace.keras
    models/unet_history.json
    models/unet_preview.png      (image | ground truth | prediction)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import json
import numpy as np
import tensorflow as tf

from synth_data import BeatBank, make_one_array
from unet import build_unet, compile_unet


def make_dataset(bank, H, W, batch, seed0):
    def gen(i):
        x, m = tf.numpy_function(lambda s: make_one_array(bank, s, H, W),
                                 [i], [tf.float32, tf.float32])
        x.set_shape((H, W, 3)); m.set_shape((H, W, 1))
        return x, m
    ds = tf.data.Dataset.range(seed0, seed0 + 10 ** 9)
    ds = ds.map(gen, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def fixed_set(bank, n, H, W, seed0=10 ** 7):
    xs, ms = zip(*[make_one_array(bank, seed0 + i, H, W) for i in range(n)])
    return np.stack(xs), np.stack(ms)


def save_preview(model, Xv, Mv, path, k=4):
    import cv2
    pred = model.predict(Xv[:k], verbose=0)[..., 0]
    rows = []
    for i in range(k):
        img = (Xv[i] * 255).astype(np.uint8)
        gt = img.copy(); gt[Mv[i, ..., 0] > .5] = (0, 0, 255)
        pr = img.copy(); pr[pred[i] > .5] = (0, 0, 255)
        rows.append(np.vstack([img, gt, pr]))
    cv2.imwrite(path, np.hstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="aami_dataset.npz",
                    help="aami_dataset.npz, or 'synthetic' for a smoke test")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=150, help="steps per epoch")
    ap.add_argument("--val_images", type=int, default=160)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--out", default="models/unet_trace.keras")
    ap.add_argument("--history", default="models/unet_history.json")
    ap.add_argument("--preview", default="models/unet_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None,
                    help="fine-tune from this .keras file (e.g. models/unet_trace.keras)")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    print("GPU devices:", tf.config.list_physical_devices("GPU") or "NONE (CPU run)")
    tf.keras.utils.set_random_seed(a.seed)
    if a.bank == "synthetic":
        tr_bank, va_bank = BeatBank.synthetic(seed=1), BeatBank.synthetic(seed=2)
        label = "SYNTHETIC beats (smoke test)"
    else:
        tr_bank = BeatBank.from_npz(a.bank, splits=(0,))
        va_bank = BeatBank.from_npz(a.bank, splits=(1,))
        label = "real MIT-BIH beats rendered onto SYNTHETIC images"
    print("beat bank:", label)

    if a.init:
        model = tf.keras.models.load_model(a.init, compile=False)
        print("fine-tuning from", a.init)
    else:
        model = build_unet(a.height, a.width, base=a.base)
    model = compile_unet(model, lr=a.lr)
    print(f"U-Net parameters: {model.count_params():,}")
    train_ds = make_dataset(tr_bank, a.height, a.width, a.batch, seed0=a.seed * 10 ** 6)
    Xv, Mv = fixed_set(va_bank, a.val_images, a.height, a.width)

    cbs = [tf.keras.callbacks.ModelCheckpoint(a.out, monitor="val_dice_coef",
                                              mode="max", save_best_only=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", mode="max",
                                                factor=0.5, patience=3, min_lr=1e-5),
           tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max",
                                            patience=8, restore_best_weights=True)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    h = model.fit(train_ds, steps_per_epoch=a.steps, epochs=a.epochs,
                  validation_data=(Xv, Mv), batch_size=a.batch,
                  callbacks=cbs, verbose=2)

    best = int(np.argmax(h.history["val_dice_coef"]))
    summary = {"data": label + "; validation images are SYNTHETIC",
               "epochs_run": len(h.history["loss"]), "best_epoch": best + 1,
               "val_dice": float(h.history["val_dice_coef"][best]),
               "val_iou": float(h.history["val_iou_metric"][best]),
               "image_size": [a.height, a.width], "parameters": int(model.count_params())}
    print("\n" + json.dumps(summary, indent=2))
    with open(a.history, "w") as f:
        json.dump({"summary": summary,
                   "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}}, f, indent=2)
    model.save(a.out)
    save_preview(model, Xv, Mv, a.preview)
    print(f"Saved {a.out}, {a.history}, {a.preview}")


if __name__ == "__main__":
    main()
