"""
build_image_dataset.py   (plan C: train the classifier in the IMAGE domain)
============================================================================
The CNN+R-R classifier was trained on raw MIT-BIH windows. Signals re-read from an image look
different (staircase, smoothed R spikes, shifted baseline), and the Day 6 ablation showed the
classifier collapses on them (macro-F1 0.70 raw vs 0.33 image). This script builds a training
set that looks like what the classifier will see in the demo:

    real MIT-BIH segment (4-10 s, random) -> rendered image (random paper/noise/surround/tilt)
    -> trace read back (mask tracker, same post-processing as digitize_unet) -> R peaks DETECTED
    (beat_detect.py, no ground-truth peaks) -> matched to the true beat labels
    -> 200-sample window + the 5 R-R features from the DETECTED peaks.

Output uses the same keys as aami_dataset.npz (X, y, record, pos, rr_prev, rr_next, rr_local,
split, class_names), so the existing trainer runs on it unchanged:

    py -3.12 src/build_image_dataset.py --data_dir data/mitdb --split_from models/aami_dataset.npz
    py -3.12 src/train_arrhythmia.py --data models/image_domain.npz --use_rr \
        --out models/arrhythmia_rr_img.keras --metrics models/arrhythmia_rr_img_metrics.json \
        --confusion_png models/arrhythmia_rr_img_confusion.png \
        --label "REAL MIT-BIH beats, SYNTHETIC images (image domain), inter-patient"

Train / validation / test records are taken from the existing aami_dataset.npz (--split_from), so
the patient-wise split is identical to the raw-signal experiments. Needs the raw records of ALL
those patients in --data_dir (DS1 and DS2): run download_ds2.py --ds1 once.

HONEST LIMITS: images are synthetic renderings (not photos); the trace is read from the exact
mask, not from the U-Net (use --digitizer unet to read it with your trained U-Net: slower and
closer to the demo); only beats that the detector found AND that match a labelled beat are kept.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cv_unet"))

from aami_dataset import CLASS_NAMES, WINDOW_BEFORE, WINDOW_AFTER, SAMPLING_RATE
from synth_data import render, PRESETS
from beat_detect import detect_r
from digitize_unet import prob_to_trace
from robustness_ablation import load_raw, rr_from_peaks, match, EDGE

FS = SAMPLING_RATE
COND_NAMES = list(PRESETS) + ["random"]


def record_splits(npz_path):
    """{record: 0 train / 1 val / 2 test} from the existing aami_dataset.npz."""
    z = np.load(npz_path, allow_pickle=False)
    rec = np.array([str(r) for r in z["record"]])
    out = {}
    for r in sorted(set(rec.tolist())):
        s = np.unique(z["split"][rec == r])
        if len(s) == 1 and s[0] >= 0:
            out[r] = int(s[0])
    return out


def sample_segments(n, pos, lab, n_seg, rng, lo, hi):
    """(start, length) pairs; about half are centred near an S/V/F beat so rare classes appear."""
    ect = pos[np.isin(lab, (1, 2, 3))]
    segs = []
    for _ in range(n_seg):
        L = int(rng.uniform(lo, hi) * FS)
        if len(ect) and rng.random() < 0.5:
            s0 = int(rng.choice(ect)) - int(rng.uniform(0.15, 0.85) * L)
        else:
            s0 = int(rng.integers(0, max(n - L, 1)))
        segs.append((int(np.clip(s0, 0, n - L)), L))
    return segs


def trace_from_mask(mask, n):
    """Same column rule as digitize_unet (run tracker, centroid, sigma-1 smoothing), n samples."""
    ys, _ = prob_to_trace(mask.astype(np.float32))
    ok = np.where(~np.isnan(ys))[0]
    if len(ok) < 2:
        return None
    y = -gaussian_filter1d(np.interp(np.arange(len(ys)), ok, ys[ok]), sigma=1.0)
    return np.interp(np.linspace(ok[0], ok[-1], n), np.arange(len(y)), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mitdb")
    ap.add_argument("--split_from", default="models/aami_dataset.npz")
    ap.add_argument("--out", default="models/image_domain.npz")
    ap.add_argument("--train_segments", type=int, default=150, help="per DS1 record")
    ap.add_argument("--test_segments", type=int, default=40, help="per DS2 record")
    ap.add_argument("--seg_min", type=float, default=4.0)
    ap.add_argument("--seg_max", type=float, default=10.0)
    ap.add_argument("--digitizer", choices=["mask", "unet"], default="mask")
    ap.add_argument("--unet", default="models/unet_trace.keras")
    ap.add_argument("--records", default=None, help="comma list to restrict (testing)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    split_of = record_splits(a.split_from)
    if a.records:
        keep = set(a.records.split(","))
        split_of = {r: s for r, s in split_of.items() if r in keep}
    SPLIT_NAME = ("train", "val", "test")
    print(f"{len(split_of)} records: " + ", ".join(f"{r}:{SPLIT_NAME[s]}" for r, s in split_of.items()))

    unet_fn = None
    if a.digitizer == "unet":
        import cv2, tempfile
        from digitize_unet import digitize_ecg_image_unet
        tmp = os.path.join(tempfile.mkdtemp(), "seg.png")
        unet_fn = lambda img, n: digitize_ecg_image_unet(tmp, n, a.unet)[0] if cv2.imwrite(tmp, img) else None

    W, Y, REC, POS, RP, RN, RL, SP = [], [], [], [], [], [], [], []
    for ri, (rec, spl) in enumerate(split_of.items()):
        d = load_raw(rec, a.data_dir)
        if d is None:
            print(f"  {rec}: no MLII or missing file, skipped"); continue
        sig, pos, lab = d
        rng = np.random.default_rng([a.seed, ri])
        n_seg = a.test_segments if spl == 2 else a.train_segments
        kept = 0
        for si, (s0, L) in enumerate(sample_segments(len(sig), pos, lab, n_seg, rng, a.seg_min, a.seg_max)):
            x = sig[s0:s0 + L] - np.median(sig[s0:s0 + L])
            m = (pos >= s0) & (pos < s0 + L)
            sp_, sl = pos[m] - s0, lab[m]
            cname = COND_NAMES[int(rng.integers(len(COND_NAMES)))]
            s = render(x.copy(), rng, 256, 768, None if cname == "random" else PRESETS[cname])
            try:
                rec_sig = unet_fn(s["image"], L) if unet_fn else trace_from_mask(s["mask"], L)
            except Exception:
                rec_sig = None
            if rec_sig is None:
                continue
            rec_sig = np.asarray(rec_sig, np.float64)
            peaks = detect_r(rec_sig)
            rr = rr_from_peaks(peaks) if len(peaks) >= 2 else None
            if rr is None:
                continue
            gi = np.where((sp_ >= EDGE) & (sp_ < L - EDGE))[0]
            pairs = match(peaks, sp_[gi])                    # {gt index: detection index}
            det_to_gt = {v: k for k, v in pairs.items()}
            for di, p in enumerate(peaks):
                if di not in det_to_gt or not (EDGE <= p < L - EDGE):
                    continue
                label = int(sl[gi[det_to_gt[di]]])
                if label < 0:
                    continue                                 # unscored beat type (Q)
                W.append(rec_sig[p - WINDOW_BEFORE:p + WINDOW_AFTER].astype(np.float32))
                Y.append(label); REC.append(rec); POS.append(s0 + p); SP.append(spl)
                RP.append(rr[0][di]); RN.append(rr[1][di]); RL.append(rr[2][di]); kept += 1
        print(f"  {rec} ({SPLIT_NAME[spl]}): {kept} beats from {n_seg} segments", flush=True)

    X = np.stack(W); y = np.array(Y, np.int64); sp = np.array(SP, np.int8)
    np.savez_compressed(a.out, X=X, y=y, record=np.array(REC), pos=np.array(POS, np.int64),
                        rr_prev=np.array(RP, np.float32), rr_next=np.array(RN, np.float32),
                        rr_local=np.array(RL, np.float32), split=sp,
                        class_names=np.array(CLASS_NAMES))
    print(f"\nSaved {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)  X{X.shape}")
    print(f"{'split':<7}{'beats':>8}" + "".join(f"{c:>8}" for c in CLASS_NAMES))
    for name, k in (("train", 0), ("val", 1), ("test", 2)):
        c = np.bincount(y[sp == k], minlength=4)
        print(f"{name:<7}{int((sp == k).sum()):>8}" + "".join(f"{int(v):>8}" for v in c))


if __name__ == "__main__":
    main()
