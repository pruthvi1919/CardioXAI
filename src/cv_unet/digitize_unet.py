"""
digitize_unet.py
=================
Drop-in replacement for image_to_signal.digitize_ecg_image that finds the
trace with the trained U-Net instead of colour thresholding.

Same return signature, so predict_pipeline.py / multi_beat_pipeline.py can
switch with a one-line import change:

    signal, col_to_index, index_to_col, trace_mask, (h, w) = \
        digitize_ecg_image_unet(path, out_len=200)

Steps: resize to the network size -> predict a trace probability map ->
per-column probability-weighted centroid of the trace -> interpolate gaps ->
smooth -> resample to out_len -> normalise (mean removed, max |x| = 1), exactly
as the classical digitizer does, so the CNN classifier sees the same scale.

Limitation: expects ONE trace per image (a single lead / rhythm strip). A
full 12-lead sheet needs lead cropping first.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

try:
    from paper_crop import find_paper_box
except ImportError:                      # imported as a package
    from .paper_crop import find_paper_box

_MODELS = {}


def _load(model_path):
    if model_path not in _MODELS:
        import tensorflow as tf
        _MODELS[model_path] = tf.keras.models.load_model(model_path, compile=False)
    return _MODELS[model_path]


def trace_probability(bgr, model_path="models/unet_trace.keras"):
    """Return (prob map at network size, (H_net, W_net))."""
    model = _load(model_path)
    _, Hn, Wn, _ = model.input_shape
    small = cv2.resize(bgr, (Wn, Hn), interpolation=cv2.INTER_AREA)
    prob = model.predict(small[None].astype(np.float32) / 255.0, verbose=0)[0, ..., 0]
    return prob, (Hn, Wn)


def prob_to_trace(prob, thr=0.5, margin_frac=0.015, min_area=40, max_jump_frac=0.25,
                  peak_mode="centroid", long_run=6):
    """
    Turn a probability map into one y value per column, robust to stray blobs.
      1. binarise, zero a thin border (printed frame / crop edge is not trace)
      2. drop tiny connected components
      3. per column, if several vertical runs exist (trace + stray bar), keep the
         run closest to the previous column's y (continuity); a column whose best
         run is both far away AND short (isolated blob) is left empty and interpolated.
    peak_mode="centroid": y = weighted centre of the chosen run (default; best vs noise,
        but flattens tall narrow QRS spikes because the run's centre is half-way up).
    peak_mode="extreme": inside a long vertical run (> long_run px) take the END of the
        run farther from the previous point, i.e. follow the line through the spike so the
        R peak keeps its height. Short runs (flat segments, slow waves) stay centroid.
    Returns ys (len W, NaN where no trace), binary mask actually used.
    """
    H, W = prob.shape
    b = (prob > thr).astype(np.uint8)
    mx, my = max(2, int(round(margin_frac * W))), max(1, int(round(margin_frac * H)))
    b[:, :mx] = 0; b[:, W - mx:] = 0; b[:my] = 0; b[H - my:] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            b[lab == k] = 0
    ys = np.full(W, np.nan)
    if b.sum() == 0:
        return ys, b
    prev = float(np.median(np.where(b > 0)[0]))
    used = np.zeros_like(b)
    wts = np.where(b > 0, prob, 0.0)
    rows = np.arange(H)
    for x in range(W):
        col = b[:, x] > 0
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]      # [start, end)
        dist = np.where(prev < starts, starts - prev, np.where(prev >= ends, prev - (ends - 1), 0.0))
        k = int(np.argmin(dist - 1e-3 * (ends - starts)))             # ties -> longer run
        # real QRS strokes can jump the full image height between columns, so only
        # reject FAR runs that are also SHORT (an isolated stray blob)
        if dist[k] > max_jump_frac * H and (ends[k] - starts[k]) < max(12, 0.05 * H):
            continue
        sl = slice(starts[k], ends[k])
        w = wts[sl, x]
        if peak_mode == "extreme" and (ends[k] - starts[k]) > long_run:
            top, bot = float(starts[k]), float(ends[k] - 1)
            ys[x] = top if abs(top - prev) >= abs(bot - prev) else bot
        else:
            ys[x] = float((w * rows[sl]).sum() / w.sum())
        used[sl, x] = 1
        prev = ys[x]
    return ys, used


def digitize_ecg_image_unet(image_path, out_len=200,
                            model_path="models/unet_trace.keras", thr=0.5, auto_crop=True,
                            peak_mode="centroid"):
    """
    auto_crop=True first finds the ECG paper in the photo (paper_crop.py) so a dark
    desk / screen surround never reaches the network. Returned mask and the two
    index<->column functions are in ORIGINAL image coordinates either way.
    """
    bgr_full = cv2.imread(image_path)
    if bgr_full is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = bgr_full.shape[:2]
    x0, y0, x1, y1 = find_paper_box(bgr_full) if auto_crop else (0, 0, w, h)
    bgr = bgr_full[y0:y1, x0:x1]
    ch, cw = bgr.shape[:2]

    prob, (Hn, Wn) = trace_probability(bgr, model_path)
    ys, used = prob_to_trace(prob, thr, peak_mode=peak_mode)
    ok = ~np.isnan(ys)
    if ok.sum() < 2:
        raise ValueError("U-Net found no ECG trace in this image.")
    xs_ok = np.where(ok)[0]
    ys_full = np.interp(np.arange(Wn), xs_ok, ys[ok])   # edges hold the nearest value

    signal_raw = -gaussian_filter1d(ys_full, sigma=1.0)
    signal = interp1d(np.linspace(0, 1, Wn), signal_raw)(np.linspace(0, 1, out_len))
    signal = signal - np.mean(signal)
    signal = signal / (np.max(np.abs(signal)) or 1.0)

    mask_full = np.zeros((h, w), np.uint8)
    mask_full[y0:y1, x0:x1] = cv2.resize(used.astype(np.uint8) * 255, (cw, ch),
                                         interpolation=cv2.INTER_NEAREST)

    def col_to_index(x_pixel):
        return int(np.clip((x_pixel - x0) / cw * out_len, 0, out_len - 1))

    def index_to_col(sig_idx):
        return int(np.clip(x0 + sig_idx / out_len * cw, 0, w - 1))

    return signal, col_to_index, index_to_col, mask_full, (h, w)


if __name__ == "__main__":
    import sys
    sig, *_rest, shape = digitize_ecg_image_unet(sys.argv[1])
    print("signal:", sig.shape, "image:", shape, "first 8:", np.round(sig[:8], 3))
