"""
beat_detect.py
===============
R-peak (QRS) detector for a 1-D signal at 360 Hz. Works on raw ECG AND on signals read back
from an image (staircase / smoothed R spikes).

Method: squared-derivative energy -> candidate peaks -> T-/P-wave suppression -> refine to the
largest deflection.
  * Suppression: a candidate that lies within `win` s of a STRONGER candidate and has less than
    `ratio` of its energy is dropped (T waves, P waves and ripple are weak next to a QRS, while
    two real beats close together, e.g. a ventricular run, have similar energy and both stay).
  * Image-derived signals need this: re-reading an image shrinks tall R spikes, so T waves look
    relatively larger and the old plain energy threshold picked them up (T waves were ~80 % of
    the false detections).

Tuned on MIT-BIH DS2 records 200, 202, 210 (win 0.36 s, ratio 0.5) and checked on held-out
records 221, 233: sensitivity 0.96-1.00, precision 0.995-1.000, HR error about 1 bpm.
Only 5 records were used. Check it on more records before quoting a number in the report.
"""
import numpy as np
from scipy.signal import find_peaks

FS = 360


def detect_r(x, fs=FS, win=0.36, ratio=0.5, floor=0.04):
    x = np.asarray(x, np.float64) - np.median(x)
    d = np.diff(x, prepend=x[0])
    w = max(3, int(0.06 * fs))
    e = np.convolve(d * d, np.ones(w) / w, mode="same")
    if e.max() <= 0:
        return np.array([], dtype=np.int64)
    pk, _ = find_peaks(e, distance=int(0.12 * fs), height=floor * e.max())
    if len(pk) == 0:
        return np.array([], dtype=np.int64)
    en = e[pk]
    kept = []
    for i in np.argsort(-en):                          # strongest first
        if all(not (abs(pk[i] - pk[j]) < win * fs and en[i] < ratio * en[j]) for j in kept):
            kept.append(i)
    pk = pk[np.array(sorted(kept))]
    r = int(0.04 * fs)                                 # refine to the largest deflection
    out = np.unique(np.array([max(0, p - r) + int(np.argmax(np.abs(x[max(0, p - r):min(len(x), p + r + 1)])))
                              for p in pk], dtype=np.int64))
    keep = [out[0]]                                    # merge refined peaks closer than 0.2 s
    for p in out[1:]:
        if p - keep[-1] < int(0.2 * fs):
            if abs(x[p]) > abs(x[keep[-1]]):
                keep[-1] = p
        else:
            keep.append(p)
    return np.array(keep, dtype=np.int64)
