"""
robustness_ablation.py   (Day 6)
================================
How much accuracy do we lose when the signal comes from an IMAGE instead of the
raw MIT-BIH samples?

For each DS2 (test-patient) record, 8-second segments of the REAL signal are
    1. rendered to an ECG image (synthetic paper/grid/noise/surround, exact trace mask),
    2. digitized again, by
         oracle    = trace read from the ground-truth mask (resolution loss only, no CV error)
         oracle_ext= same, but 'extreme' mode that keeps tall QRS spikes at full height
         unet      = your trained U-Net (+ paper auto-crop);  unet_ext = with 'extreme' mode
         classical = your HSV-threshold digitizer   (only with --classical)
    3. R peaks are DETECTED on the recovered signal (no ground-truth peaks),
    4. each detected beat gets the same 200-sample window + R-R features as in
       training and goes through the CNN+R-R classifier.
Baselines on the SAME segments:  raw_gt (raw signal, true R peaks) and
raw_detector (raw signal, our peak detector), so every drop can be attributed:
    raw_gt -> raw_detector : cost of finding peaks ourselves
    raw_detector -> oracle : cost of drawing + re-reading an image (resolution)
    oracle -> unet         : cost of the U-Net segmentation itself
    clean -> hard/realworld: cost of image degradation

Metrics are END-TO-END: a missed beat is a false negative, a false detection is a
false positive (counts against precision of whatever class it is labelled).

    python src/robustness_ablation.py --data_dir data/mitdb \
        --clf_model models/arrhythmia_rr.keras --unet models/unet_trace.keras

IMPORTANT, read before quoting numbers:
  * Beats, labels and signals are REAL (MIT-BIH, DS2). The IMAGES are SYNTHETIC
    renderings, not photographs. Photo robustness is checked separately by eye
    (try_on_image.py). Label results "real signals, synthetic images".
  * Segments are sampled with ~half containing S/V/F beats, so class balance differs
    from the full test set. Compare rows with each other, not with 0.59.
  * Segments are short, so R-R "local mean" has less context than in training. Both the
    raw and image rows see the same limit, so the DIFFERENCES are fair.
  * Beats within 0.35 s of a segment edge are not scored (window cannot fit).
  * Time scale is the known 360 Hz. Real photos need grid calibration for that.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import json
import argparse
import tempfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cv_unet"))

import train_arrhythmia as T
from aami_dataset import (AAMI_MAP, BEAT_SYMBOLS, DS2, SAMPLING_RATE,
                          WINDOW_BEFORE, WINDOW_AFTER)
from synth_data import render, PRESETS
from beat_detect import detect_r            # T-wave-suppressing QRS detector (see beat_detect.py)

FS = SAMPLING_RATE
EDGE = int(0.35 * FS)            # beats this close to a segment edge are not scored
TOL = int(0.15 * FS)             # detection-to-truth matching tolerance (samples)
NONE = 4                         # "no beat" class id used in end-to-end scoring
NAMES = ["N", "S", "V", "F", "none"]


# ----------------------------------------------------------------------------- data
def load_raw(record, data_dir):
    import wfdb
    p = os.path.join(data_dir, record)
    rec, ann = wfdb.rdrecord(p), wfdb.rdann(p, "atr")
    if "MLII" not in rec.sig_name:
        return None
    sig = rec.p_signal[:, rec.sig_name.index("MLII")].astype(np.float64)
    keep = [(s, AAMI_MAP.get(sym, -1)) for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS]
    pos = np.array([k[0] for k in keep], dtype=np.int64)
    lab = np.array([k[1] for k in keep], dtype=np.int64)         # -1 = Q / unscored
    return sig, pos, lab


def pick_segments(n, pos, lab, seg, per_record, rng):
    starts = np.arange(0, n - seg + 1, seg)
    ect = np.array([np.isin(lab[(pos >= s) & (pos < s + seg)], (1, 2, 3)).any() for s in starts])
    e_idx, n_idx = np.where(ect)[0], np.where(~ect)[0]
    k_e = min(len(e_idx), per_record // 2)
    k_n = min(len(n_idx), per_record - k_e)
    pick = np.concatenate([rng.choice(e_idx, k_e, replace=False),
                           rng.choice(n_idx, k_n, replace=False)]).astype(int)
    return starts[np.sort(pick)]


# ----------------------------------------------------------------------------- signal -> beats
def rr_from_peaks(p):
    """prev / next / local-mean R-R (samples), built exactly like aami_dataset.load_record."""
    p = np.asarray(p, np.float64)
    n = len(p)
    if n < 2:
        return None
    rr = np.diff(p)
    prev, nxt, loc = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        prev[i] = rr[i - 1] if i > 0 else rr[i]
        nxt[i] = rr[i] if i < len(rr) else rr[i - 1]
        w = rr[max(0, i - 10):i] if i > 0 else rr[:1]
        loc[i] = w.mean()
    return prev, nxt, loc


def match(det, gt, tol=TOL):
    """Greedy nearest matching. Returns {gt_index: det_index}."""
    used, pairs = set(), {}
    for gi, g in enumerate(gt):
        if len(det) == 0:
            break
        d = np.abs(det - g)
        for di in np.argsort(d):
            if d[di] > tol:
                break
            if di not in used:
                used.add(di); pairs[gi] = int(di); break
    return pairs


# ----------------------------------------------------------------------------- digitizers
def oracle_signal(mask, n):
    """Per-column mean y of the ground-truth mask -> n samples spanning the mask's columns."""
    H, W = mask.shape
    m = mask > 0
    cnt = m.sum(0)
    ok = np.where(cnt > 0)[0]
    if len(ok) < 2:
        return None
    ys = (m * np.arange(H)[:, None]).sum(0)[ok] / cnt[ok]
    cols = np.linspace(ok[0], ok[-1], n)
    return -np.interp(cols, ok, ys)


def make_digitizers(methods, unet_path):
    fns = {}
    if "oracle" in methods:
        fns["oracle"] = lambda img, mask, path, n: oracle_signal(mask, n)
    if "oracle_ext" in methods:
        from digitize_unet import prob_to_trace

        def _ext(img, mask, path, n):
            ys, _ = prob_to_trace(mask.astype(np.float32), peak_mode="extreme")
            ok = np.where(~np.isnan(ys))[0]
            return None if len(ok) < 2 else -np.interp(np.linspace(ok[0], ok[-1], n), ok, ys[ok])
        fns["oracle_ext"] = _ext
    if "unet" in methods or "unet_ext" in methods:
        from digitize_unet import digitize_ecg_image_unet
        if "unet" in methods:
            fns["unet"] = lambda img, mask, path, n: digitize_ecg_image_unet(path, n, unet_path)[0]
        if "unet_ext" in methods:
            fns["unet_ext"] = lambda img, mask, path, n: digitize_ecg_image_unet(
                path, n, unet_path, peak_mode="extreme")[0]
    if "classical" in methods:
        from image_to_signal import digitize_ecg_image
        fns["classical"] = lambda img, mask, path, n: digitize_ecg_image(path, out_len=n)[0]
    return fns


# ----------------------------------------------------------------------------- scoring
def score(rows, hr_err, model_name):
    """rows: list of (true_label [0..3 or NONE], pred_label [0..3 or NONE])."""
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    yt = np.array([r[0] for r in rows]); yp = np.array([r[1] for r in rows])
    gt_mask = yt != NONE
    matched = (yt != NONE) & (yp != NONE)
    det_total = (yp != NONE).sum()
    out = {"method": model_name, "gt_beats": int(gt_mask.sum()), "detected": int(det_total),
           "det_sensitivity": float(matched.sum() / max(gt_mask.sum(), 1)),
           "det_ppv": float(matched.sum() / max(det_total, 1)),
           "accuracy_on_matched": float((yt[matched] == yp[matched]).mean()) if matched.any() else None}
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
    for i, c in enumerate("NSVF"):
        out[f"{c}_sens"], out[f"{c}_ppv"], out[f"{c}_support"] = float(r[i]), float(p[i]), int(s[i])
    out["macro_f1_NSV"] = float(f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0))
    out["pvc_burden_true_pct"] = float(100 * (yt == 2).sum() / max(gt_mask.sum(), 1))
    out["pvc_burden_pred_pct"] = float(100 * (yp == 2).sum() / max(det_total, 1))
    out["hr_mae_bpm"] = float(np.mean(hr_err)) if len(hr_err) else None
    return out


def classify(model, windows, rrs):
    if not windows:
        return np.zeros(0, dtype=np.int64)
    X = T.normalize_np(np.asarray(windows, np.float32))
    R = T.rr_features(np.array([r[0] for r in rrs]), np.array([r[1] for r in rrs]),
                      np.array([r[2] for r in rrs]))
    return np.argmax(model.predict(T.make_inputs(X, R), batch_size=256, verbose=0), axis=1)


def hr_from(peaks):
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / FS
    rr = rr[(rr >= 0.2) & (rr <= 3.0)]
    return 60.0 / rr.mean() if len(rr) else None


class Collector:
    """Accumulates windows/features per condition, classifies at the end."""
    def __init__(self):
        self.win, self.rr, self.truth, self.gt_only, self.hr_err = [], [], [], [], []

    def add(self, sig_hat, seg_pos, seg_lab, peaks):
        """seg_pos/seg_lab: ground truth in segment coordinates. peaks: R peaks used."""
        n = len(sig_hat)
        interior = lambda a: (a >= EDGE) & (a < n - EDGE)
        gt_i = np.where(interior(seg_pos))[0]               # includes unscored (Q, label -1) beats
        gt_p, gt_l = seg_pos[gt_i], seg_lab[gt_i]
        rr = rr_from_peaks(peaks) if len(peaks) >= 2 else None
        pairs = match(np.asarray(peaks), gt_p) if rr is not None else {}
        det_of_gt = {v: k for k, v in pairs.items()}
        for di, p in enumerate(peaks):
            if not interior(np.array([p]))[0] or rr is None:
                continue
            if p - WINDOW_BEFORE < 0 or p + WINDOW_AFTER > n:
                continue
            lbl = int(gt_l[det_of_gt[di]]) if di in det_of_gt else NONE
            if lbl == -1:
                continue                                    # matched an unscored beat: ignore
            self.win.append(sig_hat[p - WINDOW_BEFORE:p + WINDOW_AFTER])
            self.rr.append((rr[0][di], rr[1][di], rr[2][di]))
            self.truth.append(lbl)
        # ground-truth beats that were never detected (or whose window did not fit)
        found = {k for k in pairs}
        for gi in range(len(gt_p)):
            if gi not in found and gt_l[gi] >= 0:
                self.gt_only.append(int(gt_l[gi]))
        h_t, h_d = hr_from(np.sort(seg_pos[seg_lab >= -1])), hr_from(np.asarray(peaks))
        if h_t and h_d:
            self.hr_err.append(abs(h_t - h_d))

    def finish(self, model, name):
        pred = classify(model, self.win, self.rr)
        rows = list(zip(self.truth, pred.tolist())) + [(t, NONE) for t in self.gt_only]
        return score(rows, self.hr_err, name)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mitdb")
    ap.add_argument("--clf_model", default="models/arrhythmia_rr.keras")
    ap.add_argument("--unet", default="models/unet_trace.keras")
    ap.add_argument("--records", default=",".join(DS2))
    ap.add_argument("--seg_seconds", type=float, default=8.0)
    ap.add_argument("--per_record", type=int, default=12)
    ap.add_argument("--conditions", default="clean,hard,realworld")
    ap.add_argument("--methods", default="oracle,unet")
    ap.add_argument("--classical", action="store_true", help="also run the classical digitizer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="models")
    a = ap.parse_args()

    import tensorflow as tf
    clf = tf.keras.models.load_model(a.clf_model, compile=False)
    methods = a.methods.split(",") + (["classical"] if a.classical else [])
    digit = make_digitizers(methods, a.unet)
    conds = a.conditions.split(",")
    seg = int(a.seg_seconds * FS)
    cols = {"raw_gt": Collector(), "raw_detector": Collector()}
    for c in conds:
        for m in methods:
            cols[f"{m}/{c}"] = Collector()
    tmp = tempfile.mkdtemp()
    n_seg, skipped = 0, 0

    for ri, rec in enumerate(a.records.split(",")):
        d = load_raw(rec, a.data_dir)
        if d is None:
            print(f"  {rec}: no MLII, skipped"); continue
        sig, pos, lab = d
        rng = np.random.default_rng([a.seed, ri])
        starts = pick_segments(len(sig), pos, lab, seg, a.per_record, rng)
        for si, s0 in enumerate(starts):
            x = sig[s0:s0 + seg] - np.median(sig[s0:s0 + seg])
            m = (pos >= s0) & (pos < s0 + seg)
            sp, sl = pos[m] - s0, lab[m]
            n_seg += 1
            cols["raw_gt"].add(x, sp, sl, sp)                       # true peaks
            cols["raw_detector"].add(x, sp, sl, detect_r(x))        # our detector
            for ci, c in enumerate(conds):
                r = np.random.default_rng([a.seed, ri, si, ci])
                s = render(x.copy(), r, 256, 768, PRESETS[c])
                path = os.path.join(tmp, "seg.png"); cv2.imwrite(path, s["image"])
                for mname, fn in digit.items():
                    try:
                        rec_sig = fn(s["image"], s["mask"], path, seg)
                    except Exception:
                        rec_sig = None
                    if rec_sig is None:
                        cols[f"{mname}/{c}"].add(np.zeros(seg), sp, sl, np.array([], dtype=np.int64)); skipped += 1
                        continue
                    rec_sig = np.asarray(rec_sig, np.float64)
                    cols[f"{mname}/{c}"].add(rec_sig, sp, sl, detect_r(rec_sig))
        print(f"  {rec}: {len(starts)} segments done", flush=True)

    results = {k: v.finish(clf, k) for k, v in cols.items()}
    meta = {"data": "REAL MIT-BIH DS2 signals and labels; images are SYNTHETIC renderings (not photos)",
            "segments": n_seg, "seg_seconds": a.seg_seconds, "digitizer_failures": skipped,
            "note": "~half of the segments contain S/V/F beats; compare rows with each other, not with the full-test 0.59"}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "robustness_ablation.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    f2 = lambda v: "-" if v is None else f"{v:.3f}"
    lines = [f"Data: {meta['data']} | {n_seg} segments of {a.seg_seconds:g} s | digitizer failures: {skipped}",
             meta["note"], "",
             "| Condition | Det. sens | Det. PPV | Macro-F1 (N,S,V) | V sens | V PPV | S sens | PVC burden true -> pred % | HR MAE bpm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {f2(v['det_sensitivity'])} | {f2(v['det_ppv'])} | {f2(v['macro_f1_NSV'])} | "
                     f"{f2(v['V_sens'])} | {f2(v['V_ppv'])} | {f2(v['S_sens'])} | "
                     f"{v['pvc_burden_true_pct']:.1f} -> {v['pvc_burden_pred_pct']:.1f} | {f2(v['hr_mae_bpm'])} |")
    lines += ["", "Class support (true beats): " + ", ".join(
        f"{c}={results['raw_gt'][c + '_support']}" for c in "NSVF")]
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out_dir, "robustness_ablation.md"), "w") as f:
        f.write(table + "\n")
    print(f"\nSaved robustness_ablation.json / .md in {a.out_dir}/")


if __name__ == "__main__":
    main()
