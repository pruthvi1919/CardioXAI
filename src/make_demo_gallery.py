"""
make_demo_gallery.py
=====================
Builds the demo gallery for the app AND measures the whole pipeline on clean images.

    real MIT-BIH segment (unseen patient, DS2)
      -> rendered as a clean 25 mm/s ECG-paper image (pink grid, black trace)   [CLEAN synthetic render]
      -> arrhythmia_pipeline.analyze_strip()   (the exact code the app runs)
      -> compared with the true MIT-BIH beat annotations

Outputs
  <out_dir>/<scenario>.png               one image per scenario (the best example found)
  <out_dir>/gallery.json                 truth + prediction of the chosen images (the app shows both)
  <out_dir>/pipeline_eval_clean.json     metrics over ALL candidate segments, not only the chosen ones
  <out_dir>/pipeline_eval_clean.md

HONEST LABELLING
  * Beats are REAL MIT-BIH; the IMAGES are SYNTHETIC renders (not photos or scans).
  * Records are DS2 (test patients): the classifier never saw them in training.
  * About half of the candidate segments are centred on an S / V / F beat, so class mix is richer than
    in a random recording; do not read the beat counts as natural prevalence.
  * The gallery images are the BEST example per scenario (selection). The metrics file is computed over
    every candidate, so it is the number to quote.

Run from the project root:
    py -3.12 src/make_demo_gallery.py --data_dir data/mitdb --models_dir models --out_dir data/ecg_images/gallery
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "cv_unet"))

from synth_data import _draw_grid                       # same paper-grid drawing used to train the U-Net
from arrhythmia_pipeline import analyze_strip
from robustness_ablation import load_raw, match, EDGE, DS2
from rhythm_analytics import analyze as rhythm

FS = 360
H, W = 384, 1152
BG, MINOR, MAJOR, TRACE = (250, 250, 255), (185, 185, 255), (120, 120, 235), (15, 15, 15)
NAMES = ["N", "S", "V", "F"]


def render_clean(sig, seed=0):
    """Standard ECG paper, 25 mm/s, black trace. Pixel <-> sample convention of the project."""
    n = len(sig)
    rng = np.random.default_rng(seed)
    px_per_mm = W / ((n / FS) * 25.0)
    img = np.full((H, W, 3), BG, np.uint8)
    _draw_grid(img, px_per_mm, rng, MINOR, MAJOR)
    s = sig / (np.max(np.abs(sig)) + 1e-9)
    x = np.arange(n) * (W - 1) / max(n - 1, 1)
    y = 0.5 * H - s * 0.36 * H
    pts = np.round(np.stack([x, y], 1) * 16).astype(np.int32)
    cv2.polylines(img, [pts], False, TRACE, 2, cv2.LINE_AA, shift=4)
    return img


def candidate_starts(pos, lab, n, seg, per_record, rng):
    ect = pos[np.isin(lab, (1, 2, 3))]
    starts = []
    for _ in range(per_record):
        if len(ect) and rng.random() < 0.55:
            s0 = int(rng.choice(ect)) - int(rng.uniform(0.15, 0.85) * seg)
        else:
            s0 = int(rng.integers(0, max(n - seg, 1)))
        starts.append(int(np.clip(s0, 0, n - seg)))
    return sorted(set(starts))


def evaluate(result, seg_len, pos, lab):
    """Score one strip. Returns dict with per-beat outcomes."""
    gi = np.where((pos >= EDGE) & (pos < seg_len - EDGE))[0]
    gt_p, gt_l = pos[gi], lab[gi]
    keep = gt_l >= 0                                       # Q-type beats are not scored
    gt_p, gt_l = gt_p[keep], gt_l[keep]
    n = len(result["signal"])
    det = np.array([b["peak"] for b in result["beats"]], float) * seg_len / n
    pairs = match(det, gt_p) if len(det) else {}
    rows = []                                              # (true, pred) ; pred 4 = missed, 5 = uncertain
    matched = set(pairs.values())
    for g, d in pairs.items():
        b = result["beats"][d]
        pred = 5 if b["label"] == "U" else NAMES.index(b["label"])
        rows.append((int(gt_l[g]), pred))
    for g in range(len(gt_p)):
        if g not in pairs:
            rows.append((int(gt_l[g]), 4))
    false_det = [d for d in range(len(det)) if d not in matched and EDGE <= det[d] < seg_len - EDGE]
    return {"rows": rows, "false_detections": len(false_det), "gt_p": gt_p, "gt_l": gt_l, "det": det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mitdb")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="data/ecg_images/gallery")
    ap.add_argument("--records", default=",".join(DS2))
    ap.add_argument("--seg_seconds", type=float, default=8.0)
    ap.add_argument("--per_record", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tmp_dir = os.path.join(a.out_dir, "_candidates")
    os.makedirs(tmp_dir, exist_ok=True)
    seg = int(a.seg_seconds * FS)
    cands, all_rows, n_false, n_strips = [], [], 0, 0

    for ri, rec in enumerate(a.records.split(",")):
        d = load_raw(rec, a.data_dir)
        if d is None:
            print(f"  {rec}: skipped (no MLII)")
            continue
        sig, pos, lab = d
        rng = np.random.default_rng([a.seed, ri])
        for s0 in candidate_starts(pos, lab, len(sig), seg, a.per_record, rng):
            x = sig[s0:s0 + seg] - np.median(sig[s0:s0 + seg])
            m = (pos >= s0) & (pos < s0 + seg)
            sp, sl = pos[m] - s0, lab[m]
            path = os.path.join(tmp_dir, f"{rec}_{s0}.png")
            cv2.imwrite(path, render_clean(x, seed=s0))
            try:
                res = analyze_strip(path, a.models_dir, os.path.join(tmp_dir, "out"))
            except Exception as e:
                print(f"  {rec}@{s0}: pipeline error {type(e).__name__}: {e}")
                continue
            if res["mode"] != "calibrated":
                print(f"  {rec}@{s0}: not calibrated ({res['notes']})")
                continue
            ev = evaluate(res, seg, sp, sl)
            all_rows += ev["rows"]
            n_false += ev["false_detections"]
            n_strips += 1
            rows = ev["rows"]
            correct = sum(1 for t, p in rows if t == p)
            tl = ev["gt_l"]
            tr = rhythm(tl, ev["gt_p"], FS) if len(tl) >= 3 else None
            cands.append({"path": path, "record": rec, "start_s": s0 / FS, "n_true": len(rows),
                          "correct": correct, "false_det": ev["false_detections"],
                          "n_S": int((tl == 1).sum()), "n_V": int((tl == 2).sum()), "n_F": int((tl == 3).sum()),
                          "bigeminy": bool(tr and tr["V"]["bigeminy_episodes"]),
                          "couplet_run": bool(tr and (tr["V"]["couplets"] or tr["V"]["runs_ge3"])),
                          "truth": [{"t": float(p) / FS, "label": NAMES[int(l)]} for p, l in zip(ev["gt_p"], tl)],
                          "pred": [{"t": float(p) / FS, "label": b["label"]}
                                   for p, b in zip(ev["det"], res["beats"])],
                          "hr_pred": (res["rhythm"] or {}).get("heart_rate", {}).get("mean_bpm") if res["rhythm"] else None,
                          "hr_true": (tr or {}).get("heart_rate", {}) and tr["heart_rate"]["mean_bpm"]})
        print(f"  {rec}: done  (strips so far {n_strips})", flush=True)

    # ------------------------------------------------------------------ metrics over ALL candidates
    from sklearn.metrics import precision_recall_fscore_support
    t = np.array([r[0] for r in all_rows])
    p = np.array([r[1] for r in all_rows])
    metrics = {"data_source": "REAL MIT-BIH beats (DS2, unseen patients), CLEAN SYNTHETIC rendered images, "
                              "full app pipeline (U-Net + grid calibration + detector + classifier)",
               "strips": n_strips, "true_beats_scored": int(len(t)), "false_detections": int(n_false)}
    detected = p != 4
    metrics["beat_detection_sensitivity"] = float(detected.mean())
    metrics["uncertain_share_of_detected"] = float((p[detected] == 5).mean()) if detected.any() else None
    per = {}
    for ci, c in enumerate(["N", "S", "V"]):
        tp = int(((t == ci) & (p == ci)).sum())
        fn = int(((t == ci) & (p != ci)).sum())
        fp = int(((t != ci) & (p == ci)).sum())
        per[c] = {"support": int((t == ci).sum()), "sensitivity": tp / max(tp + fn, 1),
                  "precision": tp / max(tp + fp, 1)}
        per[c]["f1"] = 2 * per[c]["sensitivity"] * per[c]["precision"] / max(per[c]["sensitivity"] + per[c]["precision"], 1e-9)
    metrics["per_class"] = per
    metrics["macro_f1_NSV"] = float(np.mean([per[c]["f1"] for c in per]))
    metrics["F_beats_in_scored_set"] = int((t == 3).sum())
    json.dump(metrics, open(os.path.join(a.out_dir, "pipeline_eval_clean.json"), "w"), indent=2)
    lines = ["# Whole-pipeline evaluation on clean rendered images", "",
             f"*{metrics['data_source']}*", "",
             f"{n_strips} strips, {len(t)} scored beats, {n_false} false detections.", "",
             f"Beat detection sensitivity: {metrics['beat_detection_sensitivity']:.3f}", "",
             "| Class | Support | Sensitivity | Precision | F1 |", "|---|---|---|---|---|"]
    for c, v in per.items():
        lines.append(f"| {c} | {v['support']} | {v['sensitivity']:.3f} | {v['precision']:.3f} | {v['f1']:.3f} |")
    lines += ["", f"Macro-F1 (N, S, V): {metrics['macro_f1_NSV']:.3f}",
              f"Beats labelled F in the scored set: {metrics['F_beats_in_scored_set']} (class F is not predicted by the app)."]
    open(os.path.join(a.out_dir, "pipeline_eval_clean.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    # ------------------------------------------------------------------ pick the gallery
    def score(c):
        return (c["correct"] / max(c["n_true"], 1)) - 0.1 * c["false_det"]

    ok = [c for c in cands if c["n_true"] >= 6]
    scenarios = {
        "01_normal_rhythm": ("Normal sinus rhythm", [c for c in ok if c["n_S"] == 0 and c["n_V"] == 0 and c["n_F"] == 0]),
        "02_isolated_pvcs": ("Isolated PVCs", [c for c in ok if 1 <= c["n_V"] <= 3 and c["n_S"] == 0 and c["n_F"] == 0]),
        "03_pvc_bigeminy": ("PVC bigeminy (every second beat is a PVC)", [c for c in ok if c["bigeminy"] and c["n_S"] == 0]),
        "04_pvc_couplet_or_run": ("PVC couplet / run", [c for c in ok if c["couplet_run"] and c["n_S"] == 0]),
        "05_pacs": ("Premature supraventricular (PAC-type) beats", [c for c in ok if c["n_S"] >= 2 and c["n_V"] == 0]),
        "06_mixed_s_and_v": ("Mixed: supraventricular and ventricular beats", [c for c in ok if c["n_S"] >= 1 and c["n_V"] >= 1]),
    }
    gallery = []
    for fname, (title, pool) in scenarios.items():
        if not pool:
            print(f"  scenario {fname}: no candidate found")
            continue
        best = max(pool, key=score)
        dest = os.path.join(a.out_dir, fname + ".png")
        cv2.imwrite(dest, cv2.imread(best["path"]))
        entry = {k: best[k] for k in ("record", "start_s", "n_true", "correct", "false_det", "truth", "pred",
                                      "hr_true", "hr_pred")}
        entry.update({"file": os.path.basename(dest), "title": title,
                      "note": "Best example of this scenario from the unseen-patient test set (real MIT-BIH beats, clean synthetic render). Average accuracy over all test strips is on the Performance tab."})
        gallery.append(entry)
        print(f"  {fname}: record {best['record']} @ {best['start_s']:.0f}s  "
              f"{best['correct']}/{best['n_true']} correct, {best['false_det']} false detections")
    json.dump(gallery, open(os.path.join(a.out_dir, "gallery.json"), "w"), indent=1)
    for f in os.listdir(tmp_dir):
        fp = os.path.join(tmp_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
    print(f"\nSaved {len(gallery)} gallery images + gallery.json + pipeline_eval_clean.* in {a.out_dir}")


if __name__ == "__main__":
    main()
