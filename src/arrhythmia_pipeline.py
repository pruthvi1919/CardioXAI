"""
arrhythmia_pipeline.py
=======================
End-to-end multi-class pipeline used by the new app (app_pro.py).

    ECG image
      -> paper auto-crop + grid calibration (px per second)          [grid_calibration.py, paper_crop.py]
      -> U-Net trace segmentation -> 1-D signal at 360 Hz            [digitize_unet.py]
      -> R-peak detection                                            [beat_detect.py]
      -> per-beat CNN + R-R features -> N / S / V (+ "uncertain")    [arrhythmia_rr_img.keras]
      -> Grad-CAM for the predicted class + rule-based reasons       [this file, xai_explain.py]
      -> rhythm analytics (heart rate, PVC count, couplets, ...)     [rhythm_analytics.py]
      -> annotated image

Two modes, chosen automatically:
  * "calibrated" : a real ECG paper grid (both directions) was found, so time is known in seconds.
                   Full 4-model pipeline above; the F output is ignored (see README of the project:
                   F is data-limited), the class is picked among N / S / V.
  * "shape-only" : no usable grid (e.g. images drawn by ecg_image_gen.py). Time is unknown, so R-R based
                   statements are not made; every beat is classified Normal vs PVC by the original
                   binary CNN (multi_beat_pipeline.py), exactly as before.

HONEST LIMITS
  * The classifier was trained on rendered images of real MIT-BIH beats, evaluated on unseen patients.
  * Beats below the confidence threshold are reported as "Uncertain" instead of a forced label.
  * Image-based PVC counts can run high (extra false V calls); no PVC burden % is shown for a short strip.
  * QRS width in the reasons is a RATIO to the strip's own confident normal beats (absolute ms is not
    reliable on image-read beats). With fewer than 2 confident normal beats it is not reported.
  * Not a medical device.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "cv_unet")):
    if p not in sys.path:
        sys.path.insert(0, p)

FS = 360
WIN_BEFORE, WIN_AFTER = 90, 110
CLASSES = ["N", "S", "V"]
CLASS_FULL = {"N": "Normal beat", "S": "Supraventricular (PAC-type) beat", "V": "Ventricular (PVC) beat",
              "U": "Uncertain"}
COLORS = {"N": "#2e9d4f", "S": "#f08c00", "V": "#d62828", "U": "#7a7a7a"}
PAPER_SPEED = 25.0

_CACHE = {}


def _model(path):
    import tensorflow as tf
    if path not in _CACHE:
        _CACHE[path] = tf.keras.models.load_model(path, compile=False)
    return _CACHE[path]


# --------------------------------------------------------------------------- helpers
def rr_from_peaks(p):
    """prev / next / local-mean R-R in samples (same construction as aami_dataset.py)."""
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


def _rr_feats(prev, nxt, loc):
    local = np.maximum(loc, 1.0)
    f = np.stack([prev / FS, nxt / FS, loc / FS, prev / local, nxt / local], axis=1)
    return np.clip(f, 0.0, 3.0).astype("float32")


def _norm_windows(W):
    W = W - W.mean(axis=1, keepdims=True)
    W = W / (np.max(np.abs(W), axis=1, keepdims=True) + 1e-8)
    return W.astype("float32")


def grad_cam_class(model, win_norm, rr, cls_index):
    """Grad-CAM of the 4-way softmax for class `cls_index` (0..3). Returns saliency (len(win)) in [0, 1]."""
    import tensorflow as tf
    x = tf.convert_to_tensor(win_norm.reshape(1, -1, 1), tf.float32)
    r = tf.convert_to_tensor(np.asarray(rr, np.float32).reshape(1, -1))
    gm = tf.keras.models.Model(model.inputs, [model.get_layer("last_conv").output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = gm([x, r])
        loss = preds[:, cls_index]
    g = tape.gradient(loss, conv)
    w = tf.reduce_mean(g, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return np.interp(np.linspace(0, 1, len(win_norm)), np.linspace(0, 1, len(cam)), cam)


def beat_reasons(label, win_norm, prev_ratio, next_ratio, prev_s, p_class, qrs_ratio=None):
    """Readable evidence for one beat. Uses the project's existing rule-based checks
    (P wave / T polarity from xai_explain.py) plus timing (premature / pause) and the
    strip-relative QRS width: qrs_ratio = this beat's QRS width / median width of the strip's
    confident normal beats (None when the strip has fewer than 2 of them)."""
    from xai_explain import clinical_reasoning, QRS_WIDE_RATIO, QRS_SIMILAR_RATIO
    c = clinical_reasoning(win_norm, WIN_BEFORE, FS)
    # The absolute ">120 ms" QRS sentence is dropped here: on image-read beats it cannot fire.
    pt_reasons = [r for r in c["reasons"] if not r.startswith("QRS complex is widened")]
    wide = qrs_ratio is not None and qrs_ratio >= QRS_WIDE_RATIO
    similar = qrs_ratio is not None and qrs_ratio < QRS_SIMILAR_RATIO
    ev = {"p_wave_present": c["p_wave_present"], "qrs_width_ms": c["qrs_width_ms"],
          "qrs_widened": c["qrs_widened"], "t_opposite": c["t_wave_opposite"],
          "qrs_ratio": None if qrs_ratio is None else float(qrs_ratio), "qrs_wide_vs_strip": bool(wide),
          "premature": bool(prev_ratio < 0.85), "pause_after": bool(next_ratio > 1.15)}
    out = []
    if label == "V":
        out.append(f"CNN: ventricular-type morphology (confidence {p_class:.0%}).")
        if wide:
            out.append(f"QRS is about {qrs_ratio:.1f}x as wide as this strip's normal beats, "
                       f"consistent with a ventricular origin.")
        out += pt_reasons
        if ev["premature"]:
            out.append(f"Premature: arrived {100 * (1 - prev_ratio):.0f}% earlier than the recent R-R average.")
        if ev["pause_after"]:
            out.append(f"Followed by a pause ({100 * (next_ratio - 1):.0f}% longer than usual), "
                       f"typical of a compensatory pause.")
        if qrs_ratio is None:
            out.append("QRS width not compared: this strip has fewer than 2 confident normal beats to use "
                       "as a reference.")
        elif similar:
            out.append(f"QRS width is similar to this strip's normal beats ({qrs_ratio:.1f}x), so this check "
                       f"does not support the PVC call; the label rests on the CNN's shape and timing evidence.")
        elif not wide:
            out.append(f"QRS is slightly wider than this strip's normal beats ({qrs_ratio:.1f}x); "
                       f"not conclusive on its own.")
        if not wide and not pt_reasons and not ev["premature"]:
            out.append("No single classic rule fired; classification rests on overall shape. "
                       "Review recommended.")
    elif label == "S":
        out.append(f"CNN: supraventricular-type morphology (confidence {p_class:.0%}).")
        if ev["premature"]:
            out.append(f"Premature: arrived {100 * (1 - prev_ratio):.0f}% earlier than the recent R-R average.")
        else:
            out.append("Timing is not clearly premature; this class is the weakest of the model.")
        if wide:
            out.append(f"QRS is about {qrs_ratio:.1f}x as wide as this strip's normal beats, which is unusual "
                       f"for a PAC-type beat (aberrant conduction or a misread PVC); review recommended.")
        elif qrs_ratio is not None:
            out.append(f"QRS is about as wide as this strip's normal beats ({qrs_ratio:.1f}x), pointing to an "
                       f"atrial rather than ventricular origin.")
        if not ev["p_wave_present"]:
            out.append("P wave not clearly visible (small waves are often lost when reading from an image).")
    elif label == "N":
        if wide:
            out.append(f"Normal-type morphology (confidence {p_class:.0%}), timing regular.")
            out.append(f"QRS is about {qrs_ratio:.1f}x this strip's typical normal width; review recommended.")
        elif qrs_ratio is not None:
            out.append(f"Normal-type morphology (confidence {p_class:.0%}); QRS width in line with this "
                       f"strip's other normal beats, timing regular.")
        else:
            out.append(f"Normal-type morphology (confidence {p_class:.0%}), timing regular.")
    else:
        out.append("Model confidence is below the threshold, so no class is asserted for this beat.")
        out.append("Best guess shown in the details; please review manually.")
        if qrs_ratio is not None:
            out.append(f"For review: QRS is {qrs_ratio:.1f}x as wide as this strip's normal beats.")
    return out, ev


def strip_qrs_reference(beats, min_ref=2):
    """Median QRS width (slope method) of the strip's confident normal beats (label 'N'),
    or None if there are fewer than min_ref of them. Floor of 20 ms guards against a bad reading."""
    ref = [b["qrs_slope_ms"] for b in beats if b["label"] == "N" and b.get("qrs_slope_ms", 0) > 0]
    if len(ref) < min_ref:
        return None
    return max(float(np.median(ref)), 20.0)


def explain_beats(beats):
    """(Re)build reasons + evidence for every beat of a calibrated strip. Call after labels change."""
    from xai_explain import qrs_width_slope_ms
    for b in beats:
        if "qrs_slope_ms" not in b:                     # results cached before this field existed
            b["qrs_slope_ms"] = qrs_width_slope_ms(b["window"], WIN_BEFORE, FS)
    ref = strip_qrs_reference(beats)
    for b in beats:
        ratio = b["qrs_slope_ms"] / ref if ref else None
        b["reasons"], b["evidence"] = beat_reasons(b["label"], b["window"], b["prev_ratio"], b["next_ratio"],
                                                   b["rr_prev_s"], b["confidence"], qrs_ratio=ratio)
        b["evidence"]["qrs_ref_ms"] = ref
    return beats


def drop_calibration_pulses(sig, peaks, fs=FS, min_plateau_s=0.14, level=0.85, low=0.30, max_flank_s=0.035):
    """Remove 'beats' that are really the square calibration pulse printed on many strips.
    A calibration pulse is a flat plateau (>= ~0.2 s) with VERTICAL sides on both ends. A heartbeat differs:
    its tip is a spike (tens of ms wide), and even a wide, rounded PVC deflection has at least one sloping side.
    A peak is dropped only if all hold: plateau at `level` of its height >= min_plateau_s, both flanks
    (level -> low) shorter than max_flank_s, and the plateau lies fully inside the strip (a plateau running
    off the right border is kept; one starting at the left border needs the stricter test below).
    Returns (kept_peaks, n_dropped)."""
    sig = np.asarray(sig, float)
    base = float(np.median(sig))
    keep, dropped = [], 0
    for p in np.asarray(peaks, int):
        a = sig[p] - base
        if abs(a) < 1e-9:
            keep.append(p)
            continue
        v = (sig - base) * np.sign(a) / abs(a)           # 1.0 at the peak, 0 at baseline
        n = len(v)
        l = p
        while l >= 0 and v[l] >= level:
            l -= 1
        r = p
        while r < n and v[r] >= level:
            r += 1
        plateau = r - l - 1
        if r >= n or plateau < min_plateau_s * fs:               # runs off the right border, or too narrow
            keep.append(p)
            continue
        if l < 0:
            # plateau starts at the LEFT border (the pulse's rising edge was trimmed away). Accept only if it
            # is long (>= 0.2 s), perfectly flat and ends with a vertical side; a real beat cut by the border
            # is shorter than that.
            rr = r
            while rr < n and v[rr] >= low:
                rr += 1
            flat = float(np.std(v[:r])) <= 0.04
            if plateau >= 0.20 * fs and flat and (rr - r) <= max_flank_s * fs:
                dropped += 1
            else:
                keep.append(p)
            continue
        ll = l
        while ll >= 0 and v[ll] >= low:
            ll -= 1
        rr = r
        while rr < n and v[rr] >= low:
            rr += 1
        if (l - ll) <= max_flank_s * fs and (rr - r) <= max_flank_s * fs:
            dropped += 1
        else:
            keep.append(p)
    return np.asarray(keep, int), dropped


# --------------------------------------------------------------------------- main entry
def analyze_strip(image_path, models_dir="models", out_dir="outputs", conf_thr=0.60,
                  clf_name="arrhythmia_rr_img.keras", unet_name="unet_trace.keras",
                  binary_name="pvc_cnn.keras", force_mode=None):
    """
    Returns a dict. Key fields: mode, beats (list), n_beats, rhythm (dict), summary (str),
    overlay_path, calibration (dict | None), signal (np.ndarray, 360 Hz), notes (list of str).
    """
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ---- 1) calibration ------------------------------------------------------
    cal = None
    mode = "shape-only"
    from paper_crop import find_paper_box
    from grid_calibration import estimate_grid
    x0, y0, x1, y1 = find_paper_box(bgr)
    crop = bgr[y0:y1, x0:x1]
    cw = x1 - x0
    try:
        g = estimate_grid(crop)
    except Exception:
        g = None
    if g is not None and len(g["axes"]) == 2 and force_mode != "shape-only":
        cal = {"px_per_mm": g["px_per_mm"], "confidence": g["confidence"], "note": g["note"],
               "px_per_s": g["px_per_mm"] * PAPER_SPEED}
        mode = "calibrated"

    unet_path = os.path.join(models_dir, unet_name)
    if mode == "calibrated":
        n = int(round(cw / cal["px_per_s"] * FS))
        if n < 600 or n > 20 * FS:                   # <1.7 s or >20 s: scale is implausible
            notes.append(f"Grid scale looked implausible ({n / FS:.1f} s across the image); "
                         f"time-based analysis switched off.")
            mode = "shape-only"
            cal = None

    if mode == "calibrated":
        from digitize_unet import digitize_ecg_image_unet
        from beat_detect import detect_r
        sig, c2i, i2c, mask, (H, W) = digitize_ecg_image_unet(image_path, out_len=n, model_path=unet_path)
        peaks = np.asarray(detect_r(sig, fs=FS), dtype=int)
        peaks, n_pulse = drop_calibration_pulses(sig, peaks)
        if n_pulse:
            notes.append(f"{n_pulse} flat-topped calibration pulse(s) were ignored (not heartbeats).")
        hr_check = 60.0 * FS / np.median(np.diff(peaks)) if len(peaks) >= 3 else None
        if hr_check is None or not (25 <= hr_check <= 230):
            notes.append("Could not find a plausible rhythm with the measured grid; "
                         "falling back to shape-only analysis.")
            mode = "shape-only"
            cal = None

    if mode == "shape-only":
        return _shape_only(image_path, models_dir, out_dir, binary_name, notes)

    # ---- 2) classify ------------------------------------------------------------
    clf = _model(os.path.join(models_dir, clf_name))
    rr = rr_from_peaks(peaks)
    RR = _rr_feats(*rr)
    valid = [i for i, p in enumerate(peaks) if p - WIN_BEFORE >= 0 and p + WIN_AFTER <= len(sig)]
    W = np.stack([sig[peaks[i] - WIN_BEFORE:peaks[i] + WIN_AFTER] for i in valid]) if valid else np.zeros((0, 200))
    Wn = _norm_windows(W)
    probs4 = clf.predict({"ecg_signal": Wn[..., None], "rr_features": RR[valid]}, verbose=0) \
        if valid else np.zeros((0, 4))
    p3 = probs4[:, :3] / np.clip(probs4[:, :3].sum(1, keepdims=True), 1e-8, None)   # F ignored

    from xai_explain import qrs_width_slope_ms
    beats = []
    for j, i in enumerate(valid):
        k = int(np.argmax(p3[j]))
        conf = float(p3[j, k])
        label = CLASSES[k] if conf >= conf_thr else "U"
        prev_ratio, next_ratio = float(RR[i, 3]), float(RR[i, 4])
        beats.append({"index": j, "peak": int(peaks[i]), "col": int(i2c(int(peaks[i]))),
                      "label": label, "best_guess": CLASSES[k], "confidence": conf,
                      "probs": {c: float(p3[j, q]) for q, c in enumerate(CLASSES)},
                      "rr_prev_s": float(RR[i, 0]), "rr_next_s": float(RR[i, 1]),
                      "prev_ratio": prev_ratio, "next_ratio": next_ratio,
                      "qrs_slope_ms": qrs_width_slope_ms(Wn[j], WIN_BEFORE, FS),
                      "window": Wn[j].copy(), "rr_feats": RR[i].copy(), "_cls_index": k})
    explain_beats(beats)        # reasons need the whole strip (QRS width is compared with its normal beats)
    skipped = len(peaks) - len(valid)
    if skipped:
        notes.append(f"{skipped} beat(s) at the very edge of the strip were not classified (incomplete window).")

    # ---- 3) rhythm analytics ----------------------------------------------------
    from rhythm_analytics import analyze, summary_text
    rhythm = None
    summary = "Not enough beats for rhythm analysis."
    if len(beats) >= 2:
        labels = [b["best_guess"] for b in beats]     # uncertain beats keep their best guess for the tally
        rhythm = analyze(labels, [b["peak"] for b in beats], FS)
        rhythm["n_uncertain"] = int(sum(b["label"] == "U" for b in beats))
        summary = summary_text(rhythm)

    result = {"mode": "calibrated", "image_path": image_path, "beats": beats, "n_beats": len(beats),
              "rhythm": rhythm, "summary": summary, "calibration": cal, "signal": sig, "peaks": peaks,
              "notes": notes, "conf_thr": conf_thr, "mask": mask, "index_to_col": i2c,
              "duration_s": len(sig) / FS}
    result["overlay_path"] = draw_overlay(result, os.path.join(
        out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_arrhythmia.png"))
    return result


def relabel(result, conf_thr, out_dir="outputs"):
    """Apply a new confidence threshold to a calibrated result WITHOUT re-running the U-Net / CNN.
    Returns the same result dict, updated (labels, reasons, rhythm summary, overlay)."""
    if result.get("mode") != "calibrated":
        return result
    from rhythm_analytics import analyze, summary_text
    for b in result["beats"]:
        k = b["_cls_index"]
        conf = b["confidence"]
        b["label"] = CLASSES[k] if conf >= conf_thr else "U"
    explain_beats(result["beats"])     # the normal-beat QRS reference depends on the labels
    result["conf_thr"] = conf_thr
    if len(result["beats"]) >= 2:
        r = analyze([b["best_guess"] for b in result["beats"]], [b["peak"] for b in result["beats"]], FS)
        r["n_uncertain"] = int(sum(b["label"] == "U" for b in result["beats"]))
        result["rhythm"], result["summary"] = r, summary_text(r)
    os.makedirs(out_dir, exist_ok=True)
    result["overlay_path"] = draw_overlay(result, os.path.join(
        out_dir, os.path.splitext(os.path.basename(result["image_path"]))[0] + "_arrhythmia.png"))
    return result


def explain_beat(result, beat_i, models_dir="models", clf_name="arrhythmia_rr_img.keras"):
    """Grad-CAM saliency (for the predicted class) of one beat of a calibrated result."""
    b = result["beats"][beat_i]
    clf = _model(os.path.join(models_dir, clf_name))
    return grad_cam_class(clf, b["window"], b["rr_feats"], b["_cls_index"])


# --------------------------------------------------------------------------- drawing
def draw_overlay(result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    bgr = cv2.imread(result["image_path"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    fig = plt.figure(figsize=(11, 11 * H / W + 0.5), dpi=110)
    ax = fig.add_axes([0, 0, 1, 0.93])
    ax.imshow(rgb)
    half = int(0.12 * result["calibration"]["px_per_s"] * 1.0) if result.get("calibration") else 40
    for b in result["beats"]:
        col = COLORS[b["label"]]
        ax.add_patch(Rectangle((b["col"] - half, 0), 2 * half, H, facecolor=col, alpha=0.16, edgecolor=col, lw=1.2))
        ax.text(b["col"], 14, b["label"] if b["label"] != "U" else "?", color="white", ha="center", va="center",
                fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))
    ax.axis("off")
    r = result.get("rhythm") or {}
    hr = (r.get("heart_rate") or {}).get("mean_bpm")
    title = f"{result['n_beats']} beats"
    if hr:
        title += f"  |  heart rate {hr:.0f} bpm"
    if r:
        title += f"  |  N {r['counts']['N']}  S {r['counts']['S']}  V {r['counts']['V']}"
    fig.text(0.01, 0.965, title, fontsize=11, fontweight="bold", va="center")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- shape-only fallback
def _shape_only(image_path, models_dir, out_dir, binary_name, notes):
    from multi_beat_pipeline import run_multi_beat_pipeline
    res = run_multi_beat_pipeline(image_path, os.path.join(models_dir, binary_name), out_dir=out_dir)
    beats = []
    for b in res["beats"]:
        lab = "V" if b["prediction"] == "PVC" else "N"
        beats.append({"index": b["beat_index"], "label": lab, "best_guess": lab,
                      "confidence": b["pvc_probability"] if lab == "V" else 1 - b["pvc_probability"],
                      "probs": {"N": 1 - b["pvc_probability"], "V": b["pvc_probability"]},
                      "reasons": b["reasons"], "window": b["signal"], "saliency": b["saliency_curve"],
                      "local_highlight_window": b["local_highlight_window"]})
    notes = list(notes) + ["No usable ECG paper grid, so time is unknown: beats are classified Normal vs PVC "
                           "from shape only, and heart rate / rhythm patterns are not reported."]
    return {"mode": "shape-only", "image_path": image_path, "beats": beats, "n_beats": len(beats),
            "n_pvc": res["num_pvc"], "rhythm": None, "calibration": None, "notes": notes,
            "summary": f"{res['num_beats']} beats analysed, {res['num_pvc']} flagged as PVC (shape only).",
            "overlay_path": res["output_image"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--conf_thr", type=float, default=0.60)
    a = ap.parse_args()
    r = analyze_strip(a.image, a.models_dir, a.out_dir, a.conf_thr)
    print("mode:", r["mode"], "| beats:", r["n_beats"])
    print(r["summary"])
    print("sequence:", " ".join(b["label"] if b["label"] != "U" else "?" for b in r["beats"]))
    for n in r["notes"]:
        print("NOTE:", n)
    print("overlay:", r["overlay_path"])
