"""
paper_crop.py
==============
Find the ECG paper inside a photo (dark desk / screen surround, browser UI, etc.)
and return its bounding box, so the U-Net only sees the paper. Classical CV, no
training: bright-region threshold -> close the dark grid/trace holes -> largest
blob -> bounding box. Crops only if the outside is much darker than the paper
(dark_ratio), so shadows on the paper itself are left alone.

    x0, y0, x1, y1 = find_paper_box(bgr)          # full image if no clear paper found

Returns the full image box when the paper already fills the frame (scans, web
strips), so it is safe to call on every input.
"""
import cv2
import numpy as np


def find_paper_box(bgr, inset=0.012, min_area_frac=0.15, min_fill=0.55, full_frac=0.97, dark_ratio=0.5):
    h, w = bgr.shape[:2]
    full = (0, 0, w, h)
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # bright pixels = paper; the dark grid lines and trace become holes -> close them
    bright = (gray > t).astype(np.uint8)
    k = max(5, int(0.02 * max(sh, sw)) | 1)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
    if n < 2:
        return full
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[j]
    if area < min_area_frac * sh * sw or area < min_fill * bw * bh:
        return full
    if bw >= full_frac * sw and bh >= full_frac * sh:
        return full                                  # paper already fills the frame
    # only crop when what lies OUTSIDE is clearly darker than the paper (a dark surround).
    # A shadow across the paper itself is not a surround and must not be cut away.
    inside = np.zeros((sh, sw), bool); inside[y:y + bh, x:x + bw] = True
    if inside.all() or (~inside).sum() < 0.02 * sh * sw:
        return full
    if np.median(gray[~inside]) >= dark_ratio * np.median(gray[inside]):
        return full
    x0, y0, x1, y1 = x / scale, y / scale, (x + bw) / scale, (y + bh) / scale
    ix, iy = inset * (x1 - x0), inset * (y1 - y0)    # step inside to drop the paper edge / frame
    return (int(round(x0 + ix)), int(round(y0 + iy)), int(round(x1 - ix)), int(round(y1 - iy)))


if __name__ == "__main__":
    import sys
    im = cv2.imread(sys.argv[1]); b = find_paper_box(im); print("box x0,y0,x1,y1 =", b, "of", im.shape[1], "x", im.shape[0])
    x0, y0, x1, y1 = b; cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "paper_crop_preview.png", im[y0:y1, x0:x1])
