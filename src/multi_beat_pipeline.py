"""
multi_beat_pipeline.py
=======================
Extends the single-beat pipeline (predict_pipeline.py) to handle a REAL
multi-beat ECG strip image -- i.e. one continuous trace showing several
consecutive heartbeats, which is what an actual captured/photographed
rhythm strip looks like (as opposed to a single pre-isolated beat).

Pipeline:
    multi-beat image
        --(image_to_signal.py, high-res)-->      one long digitized signal
        --(R-peak detection, this file)-->        list of individual beat
                                                    locations within it
        --(crop + resample, this file)-->         each beat resampled to
                                                    the model's expected
                                                    200-sample length
        --(model.py + xai_explain.py, per beat)--> PVC/Normal + explanation
                                                     for EACH beat
        --(ecg_image_gen.py)-->                    ONE output image with
                                                     highlight bands only
                                                     over PVC-flagged beats

This mirrors what a real deployment would do: a monitor/scan gives you a
strip with multiple beats, and the system must first locate each beat
before classifying it -- MIT-BIH's WINDOW_BEFORE/WINDOW_AFTER convention
(beat = 90 samples before R-peak + 110 samples after, at 360Hz) is used
as the reference proportion when cropping each beat window, so results
stay consistent with how the underlying CNN was trained.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from image_to_signal import digitize_ecg_image
from xai_explain import explain_prediction

# MIT-BIH beat convention the model was trained on (data_loader.py):
# WINDOW_BEFORE=90, WINDOW_AFTER=110, BEAT_LEN=200 -> R-peak sits at
# proportion 90/200=0.45 into each beat window.
BEFORE_FRAC = 90 / 200
AFTER_FRAC = 110 / 200


def detect_beats(signal, min_beat_frac=0.05, prominence_frac=0.2):
    """
    Find individual heartbeat (R-peak) locations within a long digitized
    strip signal, using PROMINENCE rather than absolute height.

    Why prominence, not a height threshold: a fixed height threshold
    (e.g. "count anything above 35% of the strip's tallest peak") breaks
    down whenever beats in the same strip have different amplitudes --
    e.g. a wide/tall PVC beat next to normal-sized beats. The normal
    beats can fall well below a threshold set relative to the tall
    outlier, causing them to be missed entirely (verified against real
    captured strip images during testing). Prominence instead measures
    how much each peak stands out from its own local baseline, so it
    correctly finds beats of varying amplitude within the same strip.

    prominence_frac=0.2 was chosen after testing against real captured
    strip images: each beat's T-wave is itself a legitimate, separate
    local peak (smaller than the QRS but still real) -- too low a
    prominence threshold (e.g. 0.1) lets T-waves get mistaken for a
    whole extra beat. 0.2 reliably distinguishes "T-wave bump of this
    beat" from "QRS of the next beat" across every test strip tried
    (both self-rendered and real photographed strips), with headroom
    on both sides (0.15-0.25 all gave correct counts).

    Two-pass: an initial loose pass estimates the actual beat spacing
    present in THIS strip, then a second pass re-detects using a distance
    threshold derived from that estimate -- this avoids a PVC's own wide,
    bizarre morphology (which can have two comparably-prominent
    deflections within a single beat) being mistaken for two beats.

    A final "split-interval" merge pass then explicitly catches a
    different failure mode: a small spurious peak (e.g. a beat's own
    T-wave) sitting roughly midway between two true beats, which splits
    one true R-R interval into two smaller gaps. Neither half looks
    anomalously short on its own (both can be close to the overall
    median), so a simple "smallest gap vs median" check misses this --
    but the two halves ADD UP to close to the true spacing seen
    elsewhere in the strip. Whenever two adjacent gaps sum to within 25%
    of that reference spacing, the peak between them is dropped as
    spurious. This was found and verified necessary by testing against
    real (not just synthetic) captured strip data during development.

    Returns: sorted 1D array of peak indices into `signal`.
    """
    n = len(signal)
    smoothed = gaussian_filter1d(signal.astype(np.float32), sigma=2)
    # PVCs can deflect either direction, so search on |signal|
    abs_sig = np.abs(smoothed)
    prominence = prominence_frac * (abs_sig.max() - abs_sig.min())

    # Pass 1: loose detection just to estimate typical beat spacing.
    loose_distance = max(5, int(n * min_beat_frac))
    candidate_peaks, _ = find_peaks(abs_sig, distance=loose_distance,
                                     prominence=prominence)
    if len(candidate_peaks) < 2:
        return candidate_peaks

    # Pass 2: re-detect using ~half the median spacing as the minimum
    # distance, which reliably separates true consecutive beats while
    # rejecting a beat's own secondary internal deflection.
    median_spacing = np.median(np.diff(candidate_peaks))
    refined_distance = max(loose_distance, int(median_spacing * 0.6))
    peaks, _ = find_peaks(abs_sig, distance=refined_distance,
                           prominence=prominence)

    # Pass 3: iteratively merge any remaining anomalously-close pair
    # (same complex counted twice, e.g. a wide PVC's R-peak and its own
    # deep S-trough both staying prominent enough to pass Pass 1/2).
    peaks = list(peaks)
    changed = True
    while changed and len(peaks) >= 3:
        changed = False
        gaps = np.diff(peaks)
        med_gap = np.median(gaps)
        min_gap_idx = int(np.argmin(gaps))
        if gaps[min_gap_idx] < 0.55 * med_gap:
            a, b = peaks[min_gap_idx], peaks[min_gap_idx + 1]
            weaker = a if abs_sig[a] < abs_sig[b] else b
            peaks.remove(weaker)
            changed = True

    # Pass 4: merge split intervals -- a spurious mid-cycle peak (e.g. a
    # T-wave) that splits one true R-R gap into two smaller ones that
    # individually look unremarkable, but sum back to the strip's true
    # spacing. Reference spacing is the LARGEST gap seen in the strip
    # (the most reliable single indicator of "one true, unsplit R-R
    # interval"), with a tight 15% tolerance -- loose enough to catch a
    # genuine split, tight enough not to misfire on real strips with
    # natural heart-rate variability, where two adjacent true gaps can
    # coincidentally sum close to a third unrelated gap (verified against
    # both a real captured strip with irregular timing, where this pass
    # must NOT fire, and synthetic strips with a genuine split, where it
    # must).
    changed = True
    while changed and len(peaks) >= 3:
        changed = False
        gaps = np.diff(peaks)
        if len(gaps) < 2:
            break
        ref = np.max(gaps)
        for i in range(len(gaps) - 1):
            pair_sum = gaps[i] + gaps[i + 1]
            if abs(pair_sum - ref) < 0.15 * ref:
                del peaks[i + 1]  # the spurious middle peak
                changed = True
                break

    return np.array(peaks)


def crop_and_resample_beat(signal, peak_idx, window_width, out_len=200):
    """
    Crop a window of `window_width` samples around `peak_idx` (using the
    MIT-BIH 0.45/0.55 before/after proportion), then resample it to
    exactly `out_len` samples -- this makes beat extraction robust to
    varying strip resolution / heart rate, since the CNN always expects
    a fixed 200-sample input regardless of the source strip's timing.

    Returns: (resampled_beat, crop_start, crop_end) in original signal
             index space (crop_start/crop_end needed to map results back
             for highlighting).
    """
    n = len(signal)
    before = int(window_width * BEFORE_FRAC)
    after = int(window_width * AFTER_FRAC)

    start = max(0, peak_idx - before)
    end = min(n, peak_idx + after)
    if end - start < 5:
        return None, start, end

    crop = signal[start:end]
    resample_fn = interp1d(np.linspace(0, 1, len(crop)), crop)
    resampled = resample_fn(np.linspace(0, 1, out_len))
    return resampled, start, end


def run_multi_beat_pipeline(image_path, model_path, out_dir="outputs",
                             sampling_rate=360, digitize_len=2000):
    """
    Full multi-beat pipeline: image -> per-beat PVC/Normal + explanation
    -> one annotated output image.
    """
    import os
    import tensorflow as tf
    from ecg_image_gen import signal_to_ecg_image

    os.makedirs(out_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    beat_len = model.input_shape[1]

    # 1) Digitize the WHOLE strip at high resolution (not compressed to
    #    200 samples like the single-beat path) so multiple beats and
    #    their relative timing are preserved for peak detection.
    full_signal, _, _, _, _ = digitize_ecg_image(image_path, out_len=digitize_len)

    # 2) Locate each beat (R-peak) within the strip.
    peaks = detect_beats(full_signal)
    if len(peaks) == 0:
        raise ValueError("No beats detected in this strip -- check image "
                          "quality or that it actually contains a visible "
                          "ECG trace.")

    # Estimate typical beat spacing (R-R interval) from detected peaks;
    # used as the crop window width for every beat. Falls back to a
    # sensible default if only one beat was found.
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks)
        window_width = int(np.median(rr_intervals) * 1.15)  # slight padding
    else:
        window_width = int(digitize_len * 0.15)

    # 3) Classify each detected beat independently.
    beat_results = []
    highlight_ranges = []
    for i, p in enumerate(peaks):
        resampled, c_start, c_end = crop_and_resample_beat(
            full_signal, p, window_width, out_len=beat_len)
        if resampled is None:
            continue

        norm = resampled - resampled.mean()
        norm = norm / (np.max(np.abs(norm)) + 1e-8)

        r_local = int(beat_len * BEFORE_FRAC)  # where R-peak lands post-resample
        report = explain_prediction(model, norm, r_peak_idx=r_local,
                                     sampling_rate=sampling_rate)

        result = {
            "beat_index": i,
            "prediction": report["prediction"],
            "pvc_probability": report["pvc_probability"],
            "reasons": report["reasons"],
            "strip_position": (c_start, c_end),
            "signal": norm,
            "saliency_curve": report["saliency_curve"],
            "local_highlight_window": report["highlight_window"],
        }
        beat_results.append(result)

        if report["prediction"] == "PVC":
            highlight_ranges.append((c_start, c_end))

    # 4) Render one annotated output image (highlight only PVC beats).
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}_multibeat_explained.png")

    n_pvc = sum(1 for r in beat_results if r["prediction"] == "PVC")
    title = f"{len(beat_results)} beats analyzed | {n_pvc} PVC flagged"

    signal_to_ecg_image(
        full_signal, out_path,
        highlight_range=highlight_ranges if highlight_ranges else None,
        title=title,
        fig_w=max(6, 1.2 * len(beat_results)), fig_h=2.2,
    )

    return {
        "num_beats": len(beat_results),
        "num_pvc": n_pvc,
        "beats": beat_results,
        "output_image": out_path,
    }


def print_multi_report(result):
    print("=" * 70)
    print(f"STRIP ANALYSIS: {result['num_beats']} beats detected, "
          f"{result['num_pvc']} flagged as PVC")
    print("=" * 70)
    for b in result["beats"]:
        marker = "*** PVC ***" if b["prediction"] == "PVC" else "Normal"
        print(f"  Beat {b['beat_index']+1}: {marker}  "
              f"(p={b['pvc_probability']:.3f})")
        # Print reasons whenever the beat is PVC, or when a borderline
        # review-flag was raised even though the CNN called it Normal --
        # otherwise a genuinely uncertain case (e.g. p=0.495 with strong
        # rule-based evidence) silently shows nothing, hiding exactly the
        # disagreement explain_prediction() was updated to surface.
        show_reasons = b["prediction"] == "PVC" or any(
            "NOTE" in r for r in b["reasons"])
        if show_reasons:
            for r in b["reasons"]:
                print(f"      - {r}")
    print(f"\nAnnotated output image: {result['output_image']}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/pvc_cnn.keras")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    result = run_multi_beat_pipeline(args.image, args.model, args.out_dir)
    print_multi_report(result)