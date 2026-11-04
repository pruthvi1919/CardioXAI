"""
synth_data.py
==============
Synthetic ECG "photo / scan" generator WITH EXACT TRACE MASKS, used to train
and evaluate the U-Net trace segmenter (the modern computer-vision stage).

Why synthetic: nobody hand-labels thousands of ECG photos. Instead we render
real MIT-BIH beats (or strips built from them) onto paper-like images and draw
the mask from the same polyline, so the ground truth is pixel-exact. Heavy
randomisation (grid colour/style, background, trace colour/thickness, text,
noise, JPEG, blur, shadows, tilt, perspective) is what lets the network cope
with real images it has never seen.

Everything is drawn with OpenCV (fast, no matplotlib), so thousands of images
per minute are possible on a CPU.

Pixel <-> sample convention (same as ecg_image_gen.py): sample i of an
n-sample signal sits at x = i * (W - 1) / (n - 1).

Also records the physical scale used (pixels per millimetre at the chosen paper
speed) so the grid-calibration step can be evaluated against ground truth.
"""

import numpy as np
import cv2

FS = 360  # MIT-BIH sampling rate (Hz)

# conditions used by the evaluation script
PRESETS = {
    "clean":   dict(),
    "noisy":   dict(noise=True),
    "tilted":  dict(rot=True),
    "shadow":  dict(light=True),
    "clutter": dict(clutter=True),
    "hard":    dict(noise=True, rot=True, light=True, clutter=True),
    # real-world failure modes seen on actual web/screen photos (v2)
    "frame":     dict(frame=True),
    "surround":  dict(surround=True, light=True, noise=True),
    "samecolor": dict(samecolor=True, darkgrid=True),
    "realworld": dict(frame=True, surround=True, samecolor=True, darkgrid=True,
                      noise=True, rot=True, light=True, clutter=True),
}
OLD_KEYS = ("noise", "rot", "light", "clutter")
NEW_KEYS = ("frame", "surround", "samecolor", "darkgrid")

BACKGROUNDS = [(255, 255, 255), (250, 250, 255), (245, 250, 255),
               (240, 245, 255), (255, 250, 245), (235, 240, 250)]        # BGR
GRIDS = [((185, 185, 255), (120, 120, 235)),    # pink / red   (classic)
         ((200, 200, 240), (150, 150, 220)),    # soft pink
         ((190, 235, 190), (130, 200, 130)),    # green
         ((235, 215, 190), (200, 170, 140)),    # blue-ish
         ((205, 205, 205), (150, 150, 150))]    # grey
TRACES = [(15, 15, 15), (35, 35, 35), (90, 40, 15), (45, 45, 70)]
TEXTS = ["II", "25 mm/s  10 mm/mV", "PATIENT 4821", "LEAD II", "HR 72",
         "50 mm/s", "MON-3", "REC 12:41:07"]


# --------------------------------------------------------------------------
# beat bank -> strips
# --------------------------------------------------------------------------
def _synthetic_beat(kind, rng, n=200):
    """Fake beat, ONLY for smoke tests when no real data is available."""
    t = np.linspace(-1, 1, n)
    g = lambda c, w, a: a * np.exp(-((t - c) ** 2) / (2 * w ** 2))
    if kind == 2:                                   # V
        b = g(0, .05, -.9) + g(.15, .05, .5) + g(.4, .09, -.35)
    elif kind == 3:                                 # F
        b = .5 * (g(-.55, .03, .15) + g(0, .012, 1)) + .5 * (g(0, .05, -.9) + g(.15, .05, .5))
    else:                                           # N / S
        b = g(-.55, .03, .15) + g(0, .012, 1) + g(.35, .08, .3)
    return b + rng.normal(0, .02, n)


class BeatBank:
    """Pool of 200-sample beats; builds random multi-beat strips."""

    def __init__(self, X, y):
        self.X = (X - X.mean(axis=1, keepdims=True)).astype(np.float32)
        self.y = np.asarray(y)
        self.by_class = {c: np.where(self.y == c)[0] for c in range(4)}
        self.classes = [c for c in range(4) if len(self.by_class[c])]

    @classmethod
    def from_npz(cls, path, splits=(0,), max_beats=40000, seed=0):
        z = np.load(path, allow_pickle=False)
        keep = np.isin(z["split"], splits)
        X, y = z["X"][keep], z["y"][keep]
        if len(y) > max_beats:
            idx = np.random.default_rng(seed).choice(len(y), max_beats, replace=False)
            X, y = X[idx], y[idx]
        return cls(X, y)

    @classmethod
    def synthetic(cls, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.choice(4, size=n, p=[.55, .15, .2, .1])
        return cls(np.array([_synthetic_beat(k, rng) for k in y]), y)

    def strip(self, rng, n_beats=None, return_centers=False):
        """Random multi-beat signal. With return_centers=True also returns the
        sample index of each beat's largest deflection (used by self-tests)."""
        if n_beats is None:
            n_beats = 1 if rng.random() < 0.15 else int(rng.integers(3, 9))
        probs = np.array([.45, .2, .2, .15])[self.classes]
        cls_ = rng.choice(self.classes, size=n_beats, p=probs / probs.sum())
        pieces, centers, cursor = [], [], 0
        for i, c in enumerate(cls_):
            if i:
                gap = int(rng.integers(0, 60))
                pieces.append(np.zeros(gap))
                cursor += gap
            beat = self.X[rng.choice(self.by_class[c])]
            centers.append(cursor + int(np.argmax(np.abs(beat))))
            pieces.append(beat)
            cursor += len(beat)
        sig = np.concatenate(pieces).astype(np.float64)
        n = len(sig)
        if rng.random() < 0.5:                      # baseline wander
            t = np.arange(n) / FS
            sig += (rng.uniform(0, .15) * np.max(np.abs(sig)) *
                    np.sin(2 * np.pi * rng.uniform(.2, .8) * t + rng.uniform(0, 6.28)))
        sig += rng.normal(0, rng.uniform(0, .03) * np.max(np.abs(sig)), n)
        return (sig, centers) if return_centers else sig


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def _draw_grid(img, px_per_mm, rng, minor, major):
    H, W = img.shape[:2]
    ox, oy = rng.uniform(0, 5 * px_per_mm, 2)
    major_w = int(rng.integers(1, 3))
    for k in range(int(W / px_per_mm) + 6):
        x = int(round(ox + (k - 5) * px_per_mm))
        if not 0 <= x < W:
            continue
        if k % 5 == 0:
            cv2.line(img, (x, 0), (x, H - 1), major, major_w)
        elif px_per_mm >= 2.5:
            cv2.line(img, (x, 0), (x, H - 1), minor, 1)
    for k in range(int(H / px_per_mm) + 6):
        y = int(round(oy + (k - 5) * px_per_mm))
        if not 0 <= y < H:
            continue
        if k % 5 == 0:
            cv2.line(img, (0, y), (W - 1, y), major, major_w)
        elif px_per_mm >= 2.5:
            cv2.line(img, (0, y), (W - 1, y), minor, 1)


def _clutter(img, rng, color):
    H, W = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        txt = TEXTS[int(rng.integers(len(TEXTS)))]
        scale = rng.uniform(.4, .9)
        band_top = rng.random() < .5
        y = int(rng.uniform(.06, .16) * H) if band_top else int(rng.uniform(.9, .97) * H)
        if rng.random() < 0.15:                     # sometimes overlap the trace
            y = int(rng.uniform(.3, .7) * H)
        x = int(rng.uniform(.02, .7) * W)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    if rng.random() < .5:                           # printed rule line
        y = int(rng.uniform(.03, .1) * H) if rng.random() < .5 else int(rng.uniform(.9, .98) * H)
        cv2.line(img, (0, y), (W - 1, y), color, 1)


def _frame(img, rng):
    """Black printed/scan border at the image edge (NOT part of the trace)."""
    H, W = img.shape[:2]
    c = tuple(int(v) for v in rng.integers(0, 60, 3))
    t = int(rng.integers(1, 5))
    inset = int(rng.integers(0, 4))
    if rng.random() < .6:                              # full rectangle
        cv2.rectangle(img, (inset, inset), (W - 1 - inset, H - 1 - inset), c, t)
    else:                                              # only top + bottom rules
        cv2.line(img, (0, inset), (W - 1, inset), c, t)
        cv2.line(img, (0, H - 1 - inset), (W - 1, H - 1 - inset), c, t)


def _surround(img, mask, rng):
    """Shrink the paper into the middle of a dark/grey canvas with fake UI bars
    (browser toolbar, thumbnails) around it -- what a screen photo looks like."""
    H, W = img.shape[:2]
    l, r = (rng.uniform(.02, .2, 2) * W).astype(int)
    t, b = (rng.uniform(.04, .22, 2) * H).astype(int)
    iw, ih = W - l - r, H - t - b
    base = int(rng.integers(10, 70)) if rng.random() < .8 else int(rng.integers(90, 160))
    tint = rng.integers(-8, 9, 3)
    canvas = np.clip(np.full((H, W, 3), base) + tint, 0, 255).astype(np.uint8)
    light = (200, 200, 200) if base < 100 else (30, 30, 30)
    # toolbar strip + icons + text
    cv2.rectangle(canvas, (0, 0), (W, max(t // 2, 4)), tuple(int(min(255, base + 25)) for _ in range(3)), -1)
    for _ in range(int(rng.integers(4, 14))):
        x = int(rng.uniform(0, .95) * W); y = int(rng.uniform(0, max(t * .5, 3)))
        cv2.rectangle(canvas, (x, y), (x + int(rng.integers(4, 18)), y + int(rng.integers(3, 8))),
                      tuple(int(v) for v in rng.integers(40, 255, 3)), -1)
    for _ in range(int(rng.integers(0, 4))):
        cv2.putText(canvas, TEXTS[int(rng.integers(len(TEXTS)))],
                    (int(rng.uniform(.02, .6) * W), int(rng.uniform(.03, max(t / H, .05) + .06) * H) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(.3, .6), light, 1, cv2.LINE_AA)
    if b > 6:                                          # thumbnail strip
        for k in range(int(rng.integers(3, 9))):
            x = int(rng.uniform(0, .95) * W)
            cv2.rectangle(canvas, (x, H - b + 2), (x + int(rng.integers(8, 30)), H - 1),
                          tuple(int(v) for v in rng.integers(30, 200, 3)), -1)
    if rng.random() < .4:                              # big dark/bright blob
        cv2.circle(canvas, (int(rng.uniform(0, W)), int(rng.uniform(0, H))),
                   int(rng.uniform(5, 25)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    canvas[t:t + ih, l:l + iw] = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
    m = np.zeros((H, W), np.uint8)
    m[t:t + ih, l:l + iw] = (cv2.resize(mask.astype(np.float32), (iw, ih),
                                        interpolation=cv2.INTER_AREA) > .3)
    return canvas, m


def _illumination(img, rng):
    H, W = img.shape[:2]
    gx = np.linspace(rng.uniform(.6, 1), rng.uniform(.6, 1), W)[None, :]
    gy = np.linspace(rng.uniform(.7, 1), rng.uniform(.7, 1), H)[:, None]
    return img * (gx * gy)[..., None]


def _warp(img, mask, rng, rot_deg, persp):
    H, W = img.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([W / 2, H / 2])
    dst = (src - c) @ R.T + c
    dst += rng.uniform(-persp, persp, dst.shape) * np.array([W, H])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(mask.astype(np.float32), M, (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img, (mask > 0.5).astype(np.uint8)


def render(sig, rng, H=256, W=768, cond=None):
    """
    Render one signal as a paper-like image. Returns dict with
        image (H,W,3 uint8 BGR), mask (H,W uint8 0/1), signal (the drawn
        samples), px_per_mm, paper_speed, fs
    cond: dict(noise, rot, light, clutter) booleans, or None for random.
    """
    if cond is None:
        cond = {} if rng.random() < .15 else {k: rng.random() < .5 for k in OLD_KEYS}
        cond.update({k: rng.random() < .3 for k in NEW_KEYS})   # real-world modes (v2)
    n = len(sig)
    duration = n / FS
    speed = 25 if rng.random() < .85 else 50
    px_per_mm = W / (duration * speed)

    bg = BACKGROUNDS[int(rng.integers(len(BACKGROUNDS)))]
    img = np.full((H, W, 3), bg, np.uint8)
    major = None
    if rng.random() > .1:                           # 10%: no grid at all
        minor, major = GRIDS[int(rng.integers(len(GRIDS)))]
        if cond.get("darkgrid"):                    # heavy dark major lines
            major = tuple(int(c * rng.uniform(.4, .7)) for c in major)
        _draw_grid(img, px_per_mm, rng, minor, major)

    sig_n = sig / (np.max(np.abs(sig)) + 1e-9)
    amp = rng.uniform(.5, 1.0) * .36 * H
    cy = rng.uniform(.46, .54) * H
    x = np.arange(n) * (W - 1) / max(n - 1, 1)
    y = cy - sig_n * amp
    pts = np.round(np.stack([x, y], 1) * 16).astype(np.int32)

    tcolor = TRACES[int(rng.integers(len(TRACES)))]
    if cond.get("samecolor") and major is not None:  # trace same hue as the grid
        tcolor = tuple(int(c * rng.uniform(.55, .85)) for c in major)
    if cond.get("clutter"):
        _clutter(img, rng, tcolor)                  # drawn BEFORE trace, not in mask
    cv2.polylines(img, [pts], False, tcolor, int(rng.integers(1, 4)),
                  cv2.LINE_AA, shift=4)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 1, 3, cv2.LINE_8, shift=4)

    if cond.get("frame"):
        _frame(img, rng)                            # drawn AFTER trace, not in mask
    if cond.get("surround"):
        img, mask = _surround(img, mask, rng)
    img = img.astype(np.float32)
    if cond.get("light"):
        img = _illumination(img, rng)
    if cond.get("rot"):
        img, mask = _warp(img.astype(np.uint8), mask, rng, 3.0, 0.02)
        img = img.astype(np.float32)
    if cond.get("noise"):
        s = rng.uniform(0, 1.5)
        if s > .3:
            img = cv2.GaussianBlur(img, (0, 0), s)
        img += rng.normal(0, rng.uniform(2, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if cond.get("noise"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return {"image": img, "mask": mask, "signal": sig, "px_per_mm": px_per_mm,
            "paper_speed": speed, "fs": FS}


def make_sample(bank, rng, H=256, W=768, cond=None):
    return render(bank.strip(rng), rng, H, W, cond)


def make_one_array(bank, seed, H, W):
    """(image float32 [0,1] BGR, mask float32 (H,W,1)) for tf.data."""
    rng = np.random.default_rng(int(seed))
    s = make_sample(bank, rng, H, W)
    return (s["image"].astype(np.float32) / 255.0,
            s["mask"].astype(np.float32)[..., None])


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="synthetic")
    ap.add_argument("--out_dir", default="outputs/synth_preview")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    bank = BeatBank.synthetic() if a.bank == "synthetic" else BeatBank.from_npz(a.bank)
    os.makedirs(a.out_dir, exist_ok=True)
    for i, (name, cond) in enumerate(PRESETS.items()):
        s = render(bank.strip(np.random.default_rng(i)), np.random.default_rng(100 + i), cond=cond)
        ov = s["image"].copy()
        ov[s["mask"] > 0] = (0, 0, 255)
        cv2.imwrite(os.path.join(a.out_dir, f"{name}.png"), np.vstack([s["image"], ov]))
    print("wrote previews to", a.out_dir)
