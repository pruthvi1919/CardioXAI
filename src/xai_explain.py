"""
xai_explain.py
===============
The EXPLAINABLE AI stage. Two layers of explanation, combined:

1. GRAD-CAM (model-level, "where"):
   Uses gradients flowing into the last Conv1D layer to produce a
   saliency map over time -> tells us WHICH part of the beat the CNN
   focused on to make its decision. This is the "highlight" region.

2. RULE-BASED CLINICAL REASONING (domain-level, "why", human-readable):
   Within the salient window, run simple signal heuristics that mirror
   how a cardiologist reads a PVC:
     - Absence / suppression of P wave before QRS
     - Widened QRS complex (duration > normal threshold)
     - T wave polarity opposite to the QRS main deflection
   These are reported alongside Grad-CAM so the explanation is both
   data-driven (CNN attention) and clinically grounded (readable reasons).

Together these directly satisfy your project's XAI requirement: "explain
how PVC was detected (absence of P wave / abnormal QRS / T wave inversion)".

3. STRIP-RELATIVE QRS WIDTH (qrs_width_slope_ms, used by arrhythmia_pipeline.py):
   A signal read back from an image is smoothed, so an absolute "QRS > 120 ms"
   rule does not work on it (the old +-60 ms rule can never exceed ~114 ms).
   Instead the QRS is measured the same way on every beat of the strip and
   compared with the strip's own confident normal beats. Image effects hit all
   beats equally, so the RATIO is meaningful even though the absolute ms is not.
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d


# ---------- 1. Grad-CAM for 1D CNN ----------

def grad_cam_1d(model, signal, last_conv_name="last_conv"):
    """
    signal: 1D array, length = model input length (already normalized)
    Returns: saliency (1D array, same length as signal, values in [0,1])
    """
    x = signal.reshape(1, -1, 1).astype(np.float32)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(x)
        loss = predictions[:, 0]  # PVC probability (sigmoid output)

    grads = tape.gradient(loss, conv_out)              # (1, T', C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))   # (C,)

    conv_out = conv_out[0]                              # (T', C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (T',)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    # upsample heatmap (computed at conv resolution) back to signal length
    heatmap = heatmap.numpy()
    saliency = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, len(heatmap)),
        heatmap
    )
    return saliency, float(predictions.numpy()[0, 0])


def saliency_to_window(saliency, threshold=0.5, pad_frac=0.06):
    """Collapse a saliency curve into a single contiguous (start, end)
    highlight window — the most salient contiguous region above threshold.

    The raw Grad-CAM saliency is computed at the last conv layer's coarse
    resolution (e.g. 50 timesteps for a 200-sample beat) and then upsampled,
    which makes it blocky/jagged. We smooth it first so the resulting window
    covers a coherent region instead of a narrow, arbitrarily-placed sliver,
    then pad the window slightly so it visually spans the abnormal area
    rather than just the single highest-saliency point.
    """
    n = len(saliency)
    smoothed = gaussian_filter1d(saliency.astype(np.float32), sigma=max(1, n // 40))
    smoothed = smoothed / (smoothed.max() + 1e-8)

    above = smoothed >= threshold
    if not above.any():
        idx = int(np.argmax(smoothed))
        start, end = max(0, idx - 10), min(n, idx + 10)
    else:
        # find longest contiguous run of True
        best_start, best_len, cur_start, cur_len = 0, 0, None, 0
        for i, v in enumerate(above):
            if v:
                if cur_start is None:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_start, cur_len = None, 0
        start, end = best_start, best_start + best_len

    pad = int(n * pad_frac)
    return max(0, start - pad), min(n, end + pad)


# ---------- 2. Rule-based clinical explanation ----------

def clinical_reasoning(signal, r_peak_idx, sampling_rate=360,
                        p_search_before=(0.28, 0.12),  # seconds before R peak
                        qrs_half_window=0.06,           # seconds
                        t_search_after=(0.10, 0.40)):   # seconds after R peak
    """
    signal      : normalized 1D beat, R-peak roughly centered
    r_peak_idx  : index of the R peak within `signal`
    Returns dict with booleans + human-readable reasons.
    """
    n = len(signal)

    def sec_to_samples(sec):
        return int(sec * sampling_rate)

    reasons = []

    # --- P wave check: look for a small local bump before the QRS ---
    p_start = max(0, r_peak_idx - sec_to_samples(p_search_before[0]))
    p_end = max(0, r_peak_idx - sec_to_samples(p_search_before[1]))
    p_region = signal[p_start:p_end] if p_end > p_start else np.array([0])
    p_amplitude = np.max(p_region) - np.mean(signal) if len(p_region) else 0
    qrs_amp_ref = np.max(np.abs(signal)) + 1e-8
    p_wave_present = (p_amplitude / qrs_amp_ref) > 0.08
    if not p_wave_present:
        reasons.append("No discernible P wave detected before the QRS "
                        "complex (suppressed/absent P wave is a hallmark "
                        "of a PVC's ectopic ventricular origin).")

    # --- QRS width check ---
    qrs_start_search = max(0, r_peak_idx - sec_to_samples(qrs_half_window))
    qrs_end_search = min(n, r_peak_idx + sec_to_samples(qrs_half_window))
    qrs_segment = signal[qrs_start_search:qrs_end_search]
    thresh = 0.3 * np.max(np.abs(qrs_segment)) if len(qrs_segment) else 0
    above = np.where(np.abs(qrs_segment) > thresh)[0]
    qrs_width_sec = (above[-1] - above[0]) / sampling_rate if len(above) > 1 else 0
    qrs_widened = qrs_width_sec > 0.12  # >120ms = widened by clinical convention
    if qrs_widened:
        reasons.append(f"QRS complex is widened (~{qrs_width_sec*1000:.0f} ms, "
                        f"normal < 120 ms), consistent with abnormal "
                        f"ventricular depolarization.")

    # --- T wave polarity check (opposite to QRS main deflection) ---
    qrs_main_sign = np.sign(signal[r_peak_idx])
    t_start = min(n, r_peak_idx + sec_to_samples(t_search_after[0]))
    t_end = min(n, r_peak_idx + sec_to_samples(t_search_after[1]))
    t_region = signal[t_start:t_end] if t_end > t_start else np.array([0])
    t_peak_val = t_region[np.argmax(np.abs(t_region))] if len(t_region) else 0
    t_opposite = np.sign(t_peak_val) != 0 and np.sign(t_peak_val) != qrs_main_sign
    if t_opposite:
        reasons.append("T wave is deflected opposite to the main QRS "
                        "direction (discordant T wave), typical of "
                        "abnormal ventricular repolarization in PVC.")

    return {
        "p_wave_present": bool(p_wave_present),
        "qrs_widened": bool(qrs_widened),
        "qrs_width_ms": float(qrs_width_sec * 1000),
        "t_wave_opposite": bool(t_opposite),
        "reasons": reasons,
    }


# ---------- 2b. QRS width for strip-relative comparison ----------

# Cut-offs for (beat QRS width) / (median QRS width of the strip's confident normal beats).
# Chosen on DS1 training + validation patients only (not on the DS2 test patients):
# at 1.30 about 76 % of true PVCs and 3-4 % of true normal beats are flagged as wider.
QRS_WIDE_RATIO = 1.30
QRS_SIMILAR_RATIO = 1.15


def qrs_width_slope_ms(signal, r_peak_idx, sampling_rate=360, frac=0.10, smooth=2,
                       core_ms=50, search_ms=(140, 160), hold_ms=10):
    """QRS width (ms) as the steep-slope region around the R peak.

    1. light smoothing, then |slope| of the beat
    2. steep = |slope| >= frac * (max |slope| within +-core_ms of R)
    3. start from the first/last steep sample near R (so a flat-topped R never gives 0)
       and extend outward while any slope in the next hold_ms is still steep

    The absolute value is NOT a clinical QRS duration on image-read beats (smoothing and
    stair-steps inflate it; normal beats read ~140-160 ms). Use it only as a ratio against
    other beats measured the same way in the same strip.
    """
    x = gaussian_filter1d(np.asarray(signal, dtype=float).ravel(), smooth)
    d = np.abs(np.gradient(x))
    n = len(x)
    r = int(np.clip(r_peak_idx, 0, n - 1))
    lo = max(0, r - int(search_ms[0] / 1000 * sampling_rate))
    hi = min(n - 1, r + int(search_ms[1] / 1000 * sampling_rate))
    c0 = max(lo, r - int(core_ms / 1000 * sampling_rate))
    c1 = min(hi, r + int(core_ms / 1000 * sampling_rate))
    th = frac * float(d[c0:c1 + 1].max())
    if th <= 0:
        return 0.0
    steep = np.where(d[c0:c1 + 1] >= th)[0] + c0
    left, right = int(steep[0]), int(steep[-1])
    hold = max(1, int(hold_ms / 1000 * sampling_rate))
    while left - 1 >= lo and np.any(d[max(lo, left - hold):left] >= th):
        left -= 1
    while right + 1 <= hi and np.any(d[right + 1:min(hi, right + hold) + 1] >= th):
        right += 1
    return float((right - left) / sampling_rate * 1000)


def explain_prediction(model, signal, r_peak_idx=None, sampling_rate=360):
    """
    Full XAI stage: combines Grad-CAM + rule-based reasoning into one report.
    signal: normalized beat (same preprocessing as training).
    r_peak_idx: index of R peak in `signal`; if None, assumes it's centered.
    """
    if r_peak_idx is None:
        r_peak_idx = len(signal) // 2

    saliency, pvc_prob = grad_cam_1d(model, signal)
    highlight_start, highlight_end = saliency_to_window(saliency, threshold=0.5)

    report = {
        "pvc_probability": pvc_prob,
        "prediction": "PVC" if pvc_prob >= 0.5 else "Normal",
        "highlight_window": (highlight_start, highlight_end),
        "saliency_curve": saliency,
    }

    # Rule-based reasoning is now ALWAYS computed (previously only ran when
    # pvc_prob >= 0.5), so a borderline CNN call still gets a full clinical
    # cross-check rather than silently skipping it.
    clinical = clinical_reasoning(signal, r_peak_idx, sampling_rate)
    report.update(clinical)
    n_criteria_met = sum([
        not clinical["p_wave_present"],
        clinical["qrs_widened"],
        clinical["t_wave_opposite"],
    ])

    # If the CNN's probability is genuinely uncertain (near the 0.5 decision
    # boundary) but the deterministic rule-based criteria strongly disagree
    # with its call, surface that disagreement explicitly rather than
    # silently defaulting to whichever side of 0.5 the CNN landed on --
    # this mirrors how a real clinical decision-support tool should behave:
    # flag uncertain/contested cases for review instead of hiding them.
    UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.35, 0.65
    is_uncertain = UNCERTAIN_LOW <= pvc_prob <= UNCERTAIN_HIGH

    if report["prediction"] == "PVC":
        if not clinical["reasons"]:
            report["reasons"] = [
                "Model detected an abnormal beat morphology in the "
                "highlighted region, though no single classic PVC "
                "feature (P wave / QRS width / T polarity) crossed "
                "threshold individually — flag for clinician review."
            ]
    else:
        if is_uncertain and n_criteria_met >= 2:
            report["reasons"] = clinical["reasons"] + [
                f"NOTE: model confidence is borderline (p={pvc_prob:.2f}), "
                f"but {n_criteria_met}/3 classic PVC criteria are met by "
                f"rule-based analysis — recommend clinician review despite "
                f"the Normal classification."
            ]
        else:
            # Confidently Normal: always show the plain message, even if a
            # single rule-based criterion happened to fire -- the rule-based
            # checks aren't perfectly specific, and surfacing an isolated
            # trigger here would misleadingly suggest concern the CNN
            # doesn't actually have.
            report["reasons"] = ["Beat morphology (P wave, QRS width, T wave "
                                  "polarity) falls within normal limits."]

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_dataset_demo
    from model import build_model

    X, y = load_dataset_demo(n_per_class=5)
    X_norm = X - X.mean(axis=1, keepdims=True)
    X_norm = X_norm / (np.max(np.abs(X_norm), axis=1, keepdims=True) + 1e-8)

    model = build_model(input_len=X.shape[1])
    # untrained model here just to smoke-test the XAI code path
    idx = np.where(y == 1)[0][0]
    report = explain_prediction(model, X_norm[idx])
    print("PVC probability:", report["pvc_probability"])
    print("Highlight window (samples):", report["highlight_window"])
    print("Reasons:", report["reasons"])
