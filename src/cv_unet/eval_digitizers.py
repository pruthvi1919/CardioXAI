"""
eval_digitizers.py
===================
Head-to-head: classical digitizer (HSV grid removal + threshold) vs the U-Net
digitizer, on generated test images under increasingly hard conditions.

Primary metric -- trace-extraction correlation r:
    each digitizer's per-column trace (out_len = image width, no down-sampling)
    is correlated with the ORACLE trace read from the ground-truth mask with
    the same column rule. This isolates "did the method find the trace" from
    "how much detail does the picture itself lose". A failure (no trace found /
    exception) scores 0. Also reports the share of images with r > 0.95.

Resolution ceiling (reported once, same for both methods): correlation of the
oracle trace against the signal that was originally drawn, after resampling to
200 points. It is below 1.0 because thin QRS strokes cannot be fully recovered
from a finite number of pixel columns.

    python src/cv_unet/eval_digitizers.py --bank aami_dataset.npz \
        --model models/unet_trace.keras --n 60

Test beats come from the TEST records (DS2); images use fresh random seeds
that training never saw. They are still SYNTHETIC images: this compares the
two methods under controlled corruption, it is not a real-photo benchmark.

Outputs: models/cv_eval.json, models/cv_eval.md, models/cv_eval_examples.png
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import json
import tempfile
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))          # src/ (image_to_signal.py)

from synth_data import BeatBank, render, PRESETS
from digitize_unet import digitize_ecg_image_unet
from image_to_signal import digitize_ecg_image


def norm(v):
    v = v - np.mean(v)
    return v / (np.max(np.abs(v)) or 1.0)


def oracle_trace(mask):
    """Per-column mean y of the ground-truth mask, gaps interpolated, y flipped."""
    H, W = mask.shape
    m = mask > 0
    cnt = m.sum(0)
    ok = cnt > 0
    ys = np.full(W, np.nan)
    ys[ok] = (m * np.arange(H)[:, None]).sum(0)[ok] / cnt[ok]
    xs = np.where(ok)[0]
    return -np.interp(np.arange(W), xs, ys[xs])


def corr(a, b):
    """Pearson r between two signals, resampling b to len(a)."""
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    r = np.corrcoef(norm(a), norm(b))[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="aami_dataset.npz")
    ap.add_argument("--model", default="models/unet_trace.keras")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--label", default="SYNTHETIC test images (real MIT-BIH test beats)")
    a = ap.parse_args()

    bank = BeatBank.synthetic(seed=3) if a.bank == "synthetic" \
        else BeatBank.from_npz(a.bank, splits=(2,))
    tmp = tempfile.mkdtemp()
    results, examples, ceiling = {}, None, []
    for name, cond in PRESETS.items():
        r_c, r_u = [], []
        for i in range(a.n):
            rng = np.random.default_rng(5_000_000 + i)
            s = render(bank.strip(rng), rng, a.height, a.width, cond)
            ref = oracle_trace(s["mask"])
            if name == "clean":
                ceiling.append(corr(norm(np.interp(np.linspace(0, 1, 200),
                                                   np.linspace(0, 1, len(ref)), ref)),
                                    s["signal"]))
            path = os.path.join(tmp, f"{name}_{i}.png")
            cv2.imwrite(path, s["image"])
            mc = mu = None
            try:
                sc, _, _, mc, _ = digitize_ecg_image(path, out_len=a.width)
                r_c.append(corr(sc, ref))
            except Exception:
                r_c.append(0.0)
            try:
                su, _, _, mu, _ = digitize_ecg_image_unet(path, a.width, a.model)
                r_u.append(corr(su, ref))
            except Exception:
                r_u.append(0.0)
            if name == "hard" and i == 0:
                examples = (s["image"],
                            mc if mc is not None else np.zeros(s["image"].shape[:2], np.uint8),
                            mu if mu is not None else np.zeros(s["image"].shape[:2], np.uint8))
        results[name] = {
            "classical_mean_r": float(np.mean(r_c)), "classical_r_gt_095": float(np.mean(np.array(r_c) > .95)),
            "unet_mean_r": float(np.mean(r_u)), "unet_r_gt_095": float(np.mean(np.array(r_u) > .95)),
        }
        print(f"{name:<8} classical r={results[name]['classical_mean_r']:.3f} "
              f"({results[name]['classical_r_gt_095']*100:3.0f}% >0.95)   "
              f"U-Net r={results[name]['unet_mean_r']:.3f} "
              f"({results[name]['unet_r_gt_095']*100:3.0f}% >0.95)", flush=True)

    ceil = float(np.mean(ceiling))
    lines = [f"Data: {a.label} | {a.n} images per condition | image {a.height}x{a.width}",
             f"Metric: correlation of each digitizer's trace with the oracle trace from the ground-truth mask.",
             f"Resolution ceiling (oracle trace vs drawn signal, 200 pts): r = {ceil:.3f}",
             "", "| Condition | Classical mean r | Classical % r>0.95 | U-Net mean r | U-Net % r>0.95 |",
             "|---|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['classical_mean_r']:.3f} | {v['classical_r_gt_095']*100:.0f}% | "
                     f"{v['unet_mean_r']:.3f} | {v['unet_r_gt_095']*100:.0f}% |")
    table = "\n".join(lines)
    print("\n" + table)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "cv_eval.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(a.out_dir, "cv_eval.json"), "w") as f:
        json.dump({"data": a.label, "n_per_condition": a.n,
                   "resolution_ceiling_r": ceil, "results": results}, f, indent=2)

    if examples is not None:
        img, mc, mu = examples
        H, W = img.shape[:2]

        def panel(m, title):
            p = img.copy()
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            p[m > 0] = (0, 0, 255)
            cv2.putText(p, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 120, 0), 2)
            return p
        o = img.copy()
        cv2.putText(o, "input (hard)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 120, 0), 2)
        cv2.imwrite(os.path.join(a.out_dir, "cv_eval_examples.png"),
                    np.vstack([o, panel(mc, "classical mask"), panel(mu, "U-Net mask")]))
    print(f"\nSaved cv_eval.json, cv_eval.md, cv_eval_examples.png in {a.out_dir}/")


if __name__ == "__main__":
    main()
