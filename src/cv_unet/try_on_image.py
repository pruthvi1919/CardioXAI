"""
try_on_image.py
================
Run the U-Net digitizer AND your classical digitizer on any real ECG image
(photo, scan, screenshot) and save one picture to eyeball the result:

    original | U-Net trace (red) | classical trace (blue) | digitized signals

    py -3.12 src/cv_unet/try_on_image.py --image path/to/ecg.jpg
    py -3.12 src/cv_unet/try_on_image.py --image a.jpg b.png --model models/unet_trace.keras

Output: outputs/<name>_unet_check.png for each image. Also prints how much of
the image width each method managed to trace. Works on a SINGLE-lead strip;
for a 12-lead sheet, crop one lead (e.g. the Lead II rhythm strip) first.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from digitize_unet import digitize_ecg_image_unet
from image_to_signal import digitize_ecg_image


def coverage(mask):
    """Share of image columns that contain at least one trace pixel."""
    return float((mask > 0).any(axis=0).mean())


def check(image_path, model_path, out_dir, out_len, auto_crop=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    out = {}
    try:
        su, _, _, mu, _ = digitize_ecg_image_unet(image_path, out_len, model_path, auto_crop=auto_crop)
        out["unet"] = (su, mu)
    except Exception as e:
        print(f"  U-Net failed: {e}")
    try:
        sc, _, _, mc, _ = digitize_ecg_image(image_path, out_len=out_len)
        out["classical"] = (sc, cv2.resize(mc, (w, h), interpolation=cv2.INTER_NEAREST))
    except Exception as e:
        print(f"  classical failed: {e}")

    def overlay(mask, color):
        o = rgb.copy()
        if mask is not None:
            o[cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0] = color
        return o

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), dpi=110)
    ax[0].imshow(rgb); ax[0].set_title("original", fontsize=9)
    ax[1].imshow(overlay(out["unet"][1] if "unet" in out else None, (255, 0, 0)))
    ax[1].set_title("U-Net trace (red)" + (f"  | columns traced: {coverage(out['unet'][1])*100:.0f}%"
                                           if "unet" in out else "  | FAILED"), fontsize=9)
    ax[2].imshow(overlay(out["classical"][1] if "classical" in out else None, (0, 80, 255)))
    ax[2].set_title("classical trace (blue)" + (f"  | columns traced: {coverage(out['classical'][1])*100:.0f}%"
                                                if "classical" in out else "  | FAILED"), fontsize=9)
    for a in ax[:3]:
        a.axis("off")
    if "unet" in out:
        ax[3].plot(out["unet"][0], color="red", lw=1.2, label="U-Net")
    if "classical" in out:
        ax[3].plot(out["classical"][0], color="#2255ff", lw=1.0, alpha=.8, label="classical")
    ax[3].set_title("digitized signal (normalised)", fontsize=9)
    ax[3].legend(fontsize=8, frameon=False)
    ax[3].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    path = os.path.join(out_dir, f"{name}_unet_check.png")
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--model", default="models/unet_trace.keras")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--out_len", type=int, default=768)
    ap.add_argument("--no_crop", action="store_true", help="skip automatic paper cropping")
    a = ap.parse_args()
    for p in a.image:
        print(p)
        print("  saved", check(p, a.model, a.out_dir, a.out_len, auto_crop=not a.no_crop))
