"""
test_qrs_relative.py
=====================
Checks the strip-relative QRS-width explanation (xai_explain.qrs_width_slope_ms +
arrhythmia_pipeline.explain_beats) on REAL MIT-BIH test patients (DS2, never used for training
or for choosing the 1.30 cut-off), using the image-domain beats the app's classifier sees.

    py -3.12 src/test_qrs_relative.py --models_dir models

Needs: models/image_domain.npz, models/arrhythmia_rr_img.keras.
Strips are rebuilt from the dataset: consecutive beats of one record, gaps < 2.5 s, <= 10 s long.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from xai_explain import qrs_width_slope_ms, QRS_WIDE_RATIO          # noqa: E402
import arrhythmia_pipeline as ap                                       # noqa: E402

FS = 360


def build_strips(rec, pos):
    order = np.lexsort((pos, rec))
    strip = np.empty(len(rec), int)
    sid, start = 0, None
    for k, i in enumerate(order):
        if k == 0:
            start = pos[i]
        else:
            j = order[k - 1]
            if rec[i] != rec[j] or pos[i] - pos[j] > 2.5 * FS or pos[i] - start > 10 * FS:
                sid += 1
                start = pos[i]
        strip[i] = sid
    return strip, order


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models_dir", default="models")
    a.add_argument("--thr", type=float, default=0.55)
    args = a.parse_args()
    import tensorflow as tf

    d = np.load(os.path.join(args.models_dir, "image_domain.npz"))
    m = d["split"] == 2                                                   # DS2 test patients only
    y, rec, pos = d["y"][m], d["record"][m].astype(str), d["pos"][m]
    Wn = ap._norm_windows(d["X"][m])
    RR = ap._rr_feats(d["rr_prev"][m], d["rr_next"][m], d["rr_local"][m])
    clf = tf.keras.models.load_model(os.path.join(args.models_dir, "arrhythmia_rr_img.keras"), compile=False)
    p4 = clf.predict({"ecg_signal": Wn[..., None], "rr_features": RR}, batch_size=512, verbose=0)
    p3 = p4[:, :3] / np.clip(p4[:, :3].sum(1, keepdims=True), 1e-8, None)

    # 1) the width measure never collapses to 0 ms
    widths = np.array([qrs_width_slope_ms(w, ap.WIN_BEFORE, FS) for w in Wn])
    assert (widths > 0).all(), f"{(widths <= 0).sum()} beats got a zero QRS width"
    print(f"[1] QRS width > 0 on all {len(widths)} DS2 beats (min {widths.min():.0f} ms)  OK")

    # 2) run the real explain_beats on rebuilt strips
    strip, _ = build_strips(rec, pos)
    beats_all = [None] * len(y)
    for s in np.unique(strip):
        idx = np.where(strip == s)[0]
        idx = idx[np.argsort(pos[idx])]
        beats = []
        for i in idx:
            k = int(p3[i].argmax())
            conf = float(p3[i, k])
            beats.append({"label": ap.CLASSES[k] if conf >= args.thr else "U", "confidence": conf,
                          "window": Wn[i], "prev_ratio": float(RR[i, 3]), "next_ratio": float(RR[i, 4]),
                          "rr_prev_s": float(RR[i, 0]), "_true": int(y[i])})
        ap.explain_beats(beats)
        for i, b in zip(idx, beats):
            beats_all[i] = b
    print(f"[2] explain_beats ran on {len(np.unique(strip))} strips, {len(y)} beats  OK")

    def kind(b):
        r = b["evidence"]["qrs_ratio"]
        if r is None:
            return "no reference"
        if r >= QRS_WIDE_RATIO:
            return "wider (supports V)"
        if r < 1.15:
            return "similar"
        return "slightly wider"

    print(f"\nREAL MIT-BIH test beats (DS2), image-domain waveforms, app threshold {args.thr}, F beats excluded")
    names = {0: "true N", 1: "true S", 2: "true V"}
    for lab in ("V", "S", "N"):
        print(f"\n  App label {lab}:")
        for t in (2, 0, 1):
            group = [b for b in beats_all if b["label"] == lab and b["_true"] == t]
            if not group:
                continue
            ks = [kind(b) for b in group]
            row = ", ".join(f"{k} {ks.count(k) / len(ks):.0%}" for k in
                            ("wider (supports V)", "slightly wider", "similar", "no reference") if ks.count(k))
            print(f"    {names[t]:7s} n={len(group):5d}  QRS vs strip normals: {row}")

    # old behaviour, for comparison
    from xai_explain import clinical_reasoning
    v_beats = [b for b in beats_all if b["label"] == "V"]
    old_fired = sum(clinical_reasoning(b["window"], ap.WIN_BEFORE, FS)["qrs_widened"] for b in v_beats)
    print(f"\n  Old absolute rule: 'QRS widened' fired on {old_fired}/{len(v_beats)} app-labelled V beats.")

    ex = next(b for b in beats_all if b["label"] == "V" and b["_true"] == 2 and kind(b).startswith("wider"))
    print("\n  Example reasons, a true PVC labelled V:")
    for r in ex["reasons"]:
        print("    -", r)

    # 3) relabel() still works and rebuilds the reasons when the threshold changes
    import cv2
    import tempfile
    tmp = tempfile.mkdtemp()
    img = os.path.join(tmp, "strip.png")
    cv2.imwrite(img, np.full((200, 800, 3), 255, np.uint8))
    s0 = strip[next(i for i in range(len(y)) if beats_all[i]["label"] == "V")]
    idx = np.where(strip == s0)[0]
    idx = idx[np.argsort(pos[idx])]
    bl = []
    for j, i in enumerate(idx):
        b = dict(beats_all[i])
        k = int(p3[i].argmax())
        b.update(index=j, peak=int(pos[i] - pos[idx[0]] + 100), col=50 + 60 * j, best_guess=ap.CLASSES[k],
                 _cls_index=k, probs={c: float(p3[i, q]) for q, c in enumerate(ap.CLASSES)})
        b.pop("qrs_slope_ms", None)                 # simulate a result cached before this change
        bl.append(b)
    res = {"mode": "calibrated", "image_path": img, "beats": bl, "n_beats": len(bl),
           "calibration": {"px_per_s": 100.0}, "conf_thr": args.thr}
    ap.relabel(res, 0.90, out_dir=tmp)
    assert all("qrs_ratio" in b["evidence"] for b in res["beats"])
    print(f"\n[3] relabel() at threshold 0.90 rebuilt reasons for {len(bl)} beats "
          f"(labels now {' '.join(b['label'] for b in res['beats'])})  OK")


if __name__ == "__main__":
    main()
