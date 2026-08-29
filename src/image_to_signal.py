"""
image_to_signal.py
===================
The COMPUTER VISION stage of the pipeline: takes an ECG image (PNG/JPG,
including a photo of a printed/scanned strip) and digitizes it back into
a 1D numeric signal, ready for the ML classifier.

Approach (classical CV, explainable & fast -- good for a project report):
  1. Load image, convert to grayscale.
  2. Remove the red/pink grid using color-based masking (grids are usually
     red-ish; trace is usually black/dark).
  3. Threshold to isolate the dark trace pixels.
  4. For each x-column, take the (weighted) y-position of trace pixels
     -> gives one signal sample per column.
  5. Smooth + resample to a fixed length (BEAT_LEN) so it matches what the
     model was trained on.
  6. Invert y-axis (image y grows downward, ECG amplitude grows upward) and
     normalize amplitude.

This also keeps a mapping from signal-index -> pixel-x-column, which we
reuse later to draw the XAI highlight back onto the ORIGINAL image.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def _remove_grid(bgr_img):
    """Mask out red/pink ECG grid lines, keep the black trace."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # trace = low saturation, low-to-mid value (black/dark gray)
    # grid   = reddish/pink, higher saturation
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 90, 120])
    trace_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    return trace_mask


def digitize_ecg_image(image_path, out_len=200):
    """
    Returns:
        signal        : 1D np.array of length out_len, digitized & normalized
        col_to_index   : function mapping original image x-pixel -> signal index
        trace_mask     : binary mask used (for debugging/visualization)
        img_shape      : (h, w) of the loaded image
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = bgr.shape[:2]

    # crop a small safety margin so any residual plot border/frame pixels
    # (which are NOT part of the physiological trace) don't get digitized
    margin = max(2, int(0.01 * min(h, w)))
    bgr = bgr[margin:h - margin, margin:w - margin]
    h, w = bgr.shape[:2]

    trace_mask = _remove_grid(bgr)

    # clean up small noise
    kernel = np.ones((2, 2), np.uint8)
    trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, kernel)

    ys = np.full(w, np.nan)
    for x in range(w):
        col = np.where(trace_mask[:, x] > 0)[0]
        if len(col) > 0:
            ys[x] = col.mean()  # average y of trace pixels in this column

    # interpolate over any gaps (e.g. where grid occluded the trace)
    valid = ~np.isnan(ys)
    if valid.sum() < 2:
        raise ValueError("Could not detect ECG trace in image — check "
                          "image quality or grid-removal color thresholds.")
    xs_valid = np.where(valid)[0]
    interp = interp1d(xs_valid, ys[valid], kind="linear",
                       fill_value="extrapolate")
    ys_full = interp(np.arange(w))

    # invert (image y grows down) and smooth
    signal_raw = -gaussian_filter1d(ys_full, sigma=1.5)

    # resample to fixed length
    resample_fn = interp1d(np.linspace(0, 1, w), signal_raw)
    signal = resample_fn(np.linspace(0, 1, out_len))

    # normalize amplitude to ~[-1, 1] like training data
    signal = signal - np.mean(signal)
    max_abs = np.max(np.abs(signal)) or 1.0
    signal = signal / max_abs

    def col_to_index(x_pixel):
        """Map an original image x-pixel to a signal-array index."""
        return int(np.clip(x_pixel / w * out_len, 0, out_len - 1))

    def index_to_col(sig_idx):
        """Inverse mapping: signal index -> original image x-pixel."""
        return int(np.clip(sig_idx / out_len * w, 0, w - 1))

    return signal, col_to_index, index_to_col, trace_mask, (h, w)


if __name__ == "__main__":
    signal, col2idx, idx2col, mask, shape = digitize_ecg_image(
        "outputs/test_beat.png", out_len=200)
    print("Digitized signal shape:", signal.shape)
    print("Image shape:", shape)
    print("First 10 samples:", np.round(signal[:10], 3))
