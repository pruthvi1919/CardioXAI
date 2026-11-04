"""
grid_calibration.py
====================
Reads the ECG paper grid in an image to find its physical scale, so heart
rate and R-R intervals can be computed from the picture itself.

Standard ECG paper: small box = 1 mm; at 25 mm/s one small box = 0.04 s, and
a large box (5 small) = 0.2 s. If we measure how many pixels one small box
spans (px_per_mm), then   seconds = pixels / (px_per_mm * paper_speed).

Steps:
  1. Estimate skew from the dominant gradient orientation and straighten.
  2. Take the column-wise and row-wise MEDIAN grey-level profile (a median
     ignores the trace and text, which cover few pixels per column) and
     remove slow illumination changes.
  3. Autocorrelation of each profile -> grid period; refined from several
     multiples of the period. Check for the "every 5th line is darker"
     pattern that tells small boxes from large ones.
  4. Combine horizontal and vertical estimates.

Honest limits (the tests below measure the first two):
  * Only works when grid lines are visible. No grid -> returns None.
  * If only the large 5 mm lines are visible, the spacing is read as 1 mm and
    heart rate comes out 5x too high; heart_rate() detects that when the rate
    is implausible (> 250 bpm) and corrects it. A wrongly-read very slow rate
    could slip through.
  * Images made by this project's own ecg_image_gen.py have arbitrary grid
    lines with no physical meaning; for those, use the known 360 Hz sampling
    instead of the grid.

    py -3.12 src/cv_unet/grid_calibration.py --image strip.jpg
    py -3.12 src/cv_unet/grid_calibration.py --selftest --bank aami_dataset.npz
"""

import os
import sys
import argparse
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --------------------------------------------------------------------------
def _rotate(gray, deg):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _profile(gray, axis):
    """axis 0 -> one value per column (finds vertical lines); axis 1 -> per row."""
    p = np.median(gray.astype(np.float32), axis=axis)
    p = p - gaussian_filter1d(p, sigma=max(10.0, len(p) / 20.0))   # drop shading
    return p - p.mean()


def _has_minor_lines(p, period):
    """
    Is `period` really the 5 mm (large-box) spacing, with fainter 1 mm lines
    between? Fold the profile at `period` and look for dips at period/5 steps.
    """
    B = int(round(period))
    if B < 15:
        return False
    ph = (np.arange(len(p)) / period) % 1.0
    idx = (ph * B).astype(int)
    cnt = np.bincount(idx, minlength=B)
    t = np.bincount(idx, weights=p, minlength=B) / np.maximum(cnt, 1)
    t = np.convolve(np.r_[t[-1:], t, t[:1]], [.25, .5, .25], "valid")
    i0 = int(np.argmax(np.abs(t)))
    sgn = np.sign(t[i0]) or 1.0
    major = abs(t[i0])
    q = B / 5.0

    def at(pos):
        c = int(round(pos)) % B
        return max(sgn * t[(c + d) % B] for d in (-1, 0, 1))
    on = np.mean([at(i0 + k * q) for k in (1, 2, 3, 4)])
    off = np.mean([sgn * t[int(round(i0 + (k + .5) * q)) % B] for k in (1, 2, 3)])
    return bool(on > 0.12 * major and on - off > 0.08 * major)


def _period(p, min_lag=3):
    """(period_px, strength, accent) from the autocorrelation, or None."""
    n = len(p)
    if np.std(p) < 0.4:                       # essentially flat: no grid
        return None
    ac = np.correlate(p, p, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    max_lag = n // 2
    peaks, _ = find_peaks(ac[:max_lag], prominence=0.08)
    peaks = peaks[peaks >= min_lag]
    if len(peaks) == 0:
        return None
    L1 = int(peaks[0])
    strength = float(ac[L1])
    if strength < 0.12:
        return None

    # refine step by step: each accepted multiple re-estimates the period,
    # fitted through the origin (lag at k = 0 is exactly 0)
    period = float(L1)
    ks, ls = [1], [float(L1)]
    for k in range(2, 14):
        c = int(round(k * period))
        w = max(2, int(0.15 * period))
        if c + w >= max_lag:
            break
        seg = ac[c - w:c + w + 1]
        j = int(np.argmax(seg))
        if seg[j] > 0.3 * strength:
            ks.append(k)
            ls.append(float(c - w + j))
            period = float(np.dot(ks, ls) / np.dot(ks, ks))

    minor = _has_minor_lines(p, period)
    if minor:
        period /= 5.0
    accent = bool(minor)
    if not minor and 6 * period < max_lag:    # every 5th line darker?
        def local(k):
            c = int(round(k * period)); w = max(2, int(0.1 * period))
            return float(ac[max(c - w, 0):c + w + 1].max())
        accent = local(5) - 0.5 * (local(4) + local(6)) > 0.03
    return period, strength, accent


def estimate_skew(gray, limit=8.0):
    """Skew in degrees from the dominant gradient orientation (mod 90)."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 45) % 90 - 45          # [-45, 45)
    keep = mag > np.percentile(mag, 85)
    hist, edges = np.histogram(ang[keep], bins=int(2 * limit / 0.25),
                               range=(-limit, limit), weights=mag[keep])
    if hist.sum() == 0:
        return 0.0
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def estimate_grid(image):
    """
    image: path or BGR array. Returns dict(px_per_mm, confidence, skew_deg,
    axes, note) or None if no grid is visible.
    """
    bgr = cv2.imread(image) if isinstance(image, str) else image
    if bgr is None:
        raise FileNotFoundError(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    skew = estimate_skew(gray)
    best = None
    for deg in (0.0, skew, -skew):             # keep whichever sharpens the grid most
        g = gray if deg == 0.0 else _rotate(gray, deg)
        found = {}
        for name, axis in (("x", 0), ("y", 1)):
            r = _period(_profile(g, axis))
            if r is not None:
                found[name] = r
        score = sum(v[1] for v in found.values())
        if found and (best is None or score > best[0] + 1e-9):
            best = (score, deg, found)
    if best is None:
        return None
    _, deg, found = best

    note = ""
    if len(found) == 2:
        px, py = found["x"][0], found["y"][0]
        if 0.85 < px / py < 1.18:
            period = (px * found["x"][1] + py * found["y"][1]) / (found["x"][1] + found["y"][1])
            note = "x and y agree"
        elif 4.4 < max(px, py) / min(px, py) < 5.6:
            period = min(px, py)             # one axis sees the faint 1 mm lines
            note = f"x/y differ by about 5x ({px:.1f} vs {py:.1f} px); used the smaller (1 mm)"
        else:
            k = max(found, key=lambda a: found[a][1])
            period = found[k][0]
            note = f"x/y disagree ({px:.1f} vs {py:.1f} px); used the stronger ({k})"
    else:
        k = next(iter(found))
        period = found[k][0]
        note = f"only {k}-direction lines found"
    accent = any(v[2] for v in found.values())
    return {"px_per_mm": float(period),
            "confidence": float(min(v[1] for v in found.values())),
            "skew_deg": float(deg), "axes": sorted(found), "major_lines_seen": bool(accent),
            "note": note}


def heart_rate(r_peak_cols, px_per_mm, paper_speed=25.0):
    """
    Heart rate and R-R intervals from R-peak pixel columns.
    Returns dict(hr_bpm, rr_s (array), plausible, note) or None (< 2 peaks).
    """
    r = np.sort(np.asarray(r_peak_cols, dtype=float))
    if len(r) < 2:
        return None
    px_per_s = px_per_mm * paper_speed
    rr = np.diff(r) / px_per_s
    hr = 60.0 / float(np.mean(rr))
    note = ""
    if hr > 250:                                # grid read as 1 mm was really 5 mm
        rr, hr = rr * 5, hr / 5
        note = "rate implausible; treated detected grid spacing as 5 mm"
    elif hr < 25:
        rr, hr = rr / 5, hr * 5
        note = "rate implausible; treated detected grid spacing as 0.2 mm-equivalent"
    return {"hr_bpm": hr, "rr_s": rr, "rr_mean_s": float(np.mean(rr)),
            "rr_sd_s": float(np.std(rr)), "plausible": bool(30 <= hr <= 220),
            "note": note}


# --------------------------------------------------------------------------
def selftest(bank_path, n_per_condition=25, H=256, W=768):
    """Calibrate synthetic images with KNOWN px_per_mm and beat positions."""
    from synth_data import BeatBank, render, PRESETS
    if bank_path == "synthetic":
        bank = BeatBank.synthetic(seed=5)
    else:
        b = BeatBank.from_npz(bank_path, splits=(2,))
        bank = BeatBank(b.X[b.y == 0], b.y[b.y == 0])      # N beats: clean R positions
    print(f"{'condition':<9}{'grid found':>11}{'scale err<5%':>14}{'median scale err':>18}"
          f"{'HR err<5%':>11}{'median HR err':>15}")
    tot = {"found": 0, "ok": 0, "hr_ok": 0, "n": 0}
    rows = []
    for name, cond in PRESETS.items():
        found = ok = hr_ok = 0
        e_scale, e_hr = [], []
        for i in range(n_per_condition):
            rng = np.random.default_rng(9_000_000 + i)
            sig, centers = bank.strip(rng, n_beats=int(rng.integers(3, 9)), return_centers=True)
            s = render(sig, rng, H, W, cond)
            g = estimate_grid(s["image"])
            if g is None:
                e_scale.append(1.0); e_hr.append(1.0)
                continue
            found += 1
            err = abs(g["px_per_mm"] - s["px_per_mm"]) / s["px_per_mm"]
            cols = np.array(centers) * (W - 1) / (len(sig) - 1)
            true_hr = 60.0 / (np.mean(np.diff(centers)) / s["fs"])
            est = heart_rate(cols, g["px_per_mm"], s["paper_speed"])
            # the generator gives the true paper speed; a real image would assume 25 mm/s
            herr = abs(est["hr_bpm"] - true_hr) / true_hr
            e_scale.append(err); e_hr.append(herr)
            ok += err < .05
            hr_ok += herr < .05
        rows.append((name, found, ok, hr_ok, np.median(e_scale), np.median(e_hr)))
        print(f"{name:<9}{found/n_per_condition*100:>10.0f}%{ok/n_per_condition*100:>13.0f}%"
              f"{np.median(e_scale)*100:>17.1f}%{hr_ok/n_per_condition*100:>10.0f}%"
              f"{np.median(e_hr)*100:>14.1f}%")
        tot["found"] += found; tot["ok"] += ok; tot["hr_ok"] += hr_ok; tot["n"] += n_per_condition
    print(f"\noverall: grid found {tot['found']/tot['n']*100:.0f}% | scale within 5%: "
          f"{tot['ok']/tot['n']*100:.0f}% | heart rate within 5%: {tot['hr_ok']/tot['n']*100:.0f}%")
    print("(SYNTHETIC images with known scale; paper speed taken from the generator)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bank", default="aami_dataset.npz")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.bank, a.n)
    elif a.image:
        g = estimate_grid(a.image)
        if g is None:
            print("No ECG grid found in this image: heart rate cannot be read from the "
                  "image scale. (Generated images from ecg_image_gen.py have no physical grid.)")
        else:
            print(f"grid: {g['px_per_mm']:.2f} px per mm  (1 s = {g['px_per_mm']*25:.0f} px at 25 mm/s)")
            print(f"confidence {g['confidence']:.2f} | skew {g['skew_deg']:+.2f} deg | "
                  f"{g['note']} | major lines seen: {g['major_lines_seen']}")
    else:
        ap.print_help()
