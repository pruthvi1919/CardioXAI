"""
dashboard_data.py
=================
Pre-computes the numbers behind the two evaluation dashboards in app_pro.py, from the
REAL MIT-BIH test beats (patients never used for training) in models/image_domain.npz and
the classifier the app actually runs (models/arrhythmia_rr_img.keras).

Run once from the project root (about 1-2 minutes on CPU):

    py -3.12 src/dashboard_data.py

Writes
  models/eval_test_probs.npz     class probabilities for every test beat (feeds the Evaluation Lab)
  models/xai_faithfulness.json   Grad-CAM deletion test (feeds the XAI Faithfulness tab)

Data labels: beats / labels are REAL (MIT-BIH annotations); the waveform windows are the
image-domain version (rendered image -> digitised signal), i.e. what the app really sees.
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASS_NAMES = ["N", "S", "V", "F"]
FS = 360


def load_test(models_dir):
    from train_arrhythmia import normalize_np, rr_features
    d = np.load(os.path.join(models_dir, "image_domain.npz"))
    m = d["split"] == 2
    Xn = normalize_np(d["X"][m])
    rr = rr_features(d["rr_prev"][m], d["rr_next"][m], d["rr_local"][m])
    return Xn, rr, d["y"][m].astype(int), d["record"][m].astype(str)


def batch_gradcam(model, Xn, rr, cls, bs=512):
    """Grad-CAM for class cls[i] of sample i (same maths as arrhythmia_pipeline.grad_cam_class, batched)."""
    import tensorflow as tf
    gm = tf.keras.models.Model(model.inputs, [model.get_layer("last_conv").output, model.output])
    out = []
    for s in range(0, len(Xn), bs):
        x = tf.convert_to_tensor(Xn[s:s + bs], tf.float32)
        r = tf.convert_to_tensor(rr[s:s + bs], tf.float32)
        oh = tf.one_hot(cls[s:s + bs], 4)
        with tf.GradientTape() as tape:
            conv, preds = gm([x, r])
            loss = tf.reduce_sum(preds * oh)
        g = tape.gradient(loss, conv)
        w = tf.reduce_mean(g, axis=1, keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1)).numpy()
        cam = cam / (cam.max(axis=1, keepdims=True) + 1e-8)
        n = Xn.shape[1]
        for c in cam:
            out.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(c)), c))
    return np.asarray(out, np.float32)


def best_window(cam, w, mode):
    """Start index of the contiguous window of width w with max ('top') or min ('low') total saliency."""
    cs = np.concatenate([np.zeros((cam.shape[0], 1)), np.cumsum(cam, axis=1)], axis=1)
    sums = cs[:, w:] - cs[:, :-w]
    return sums.argmax(1) if mode == "top" else sums.argmin(1)


def mask_windows(Xn, starts, w):
    X = Xn.copy()
    for i, s in enumerate(starts):
        X[i, s:s + w, 0] = 0.0                      # 0 = the beat's own mean level (inputs are mean-centred)
    return X


def faithfulness(model, Xn, rr, y, rng, fracs=(0.1, 0.2, 0.3, 0.4), n_random=5, per_class=400):
    prob = model.predict({"ecg_signal": Xn, "rr_features": rr}, batch_size=1024, verbose=0)
    pred = prob.argmax(1)
    # use beats the model gets RIGHT for N, S, V (the beats whose explanation we would show)
    idx_all = []
    for c in (0, 1, 2):
        idx = np.where((y == c) & (pred == c))[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        idx_all.append(idx)
    idx = np.concatenate(idx_all)
    X, R, Y = Xn[idx], rr[idx], y[idx]
    p0 = prob[idx, Y]
    cam = batch_gradcam(model, X, R, Y)
    n = X.shape[1]

    def conf_after(starts, w):
        Xm = mask_windows(X, starts, w)
        pm = model.predict({"ecg_signal": Xm, "rr_features": R}, batch_size=1024, verbose=0)
        return pm[np.arange(len(Y)), Y], pm.argmax(1)

    out = {"n_beats": int(len(idx)), "per_class_n": {CLASS_NAMES[c]: int((Y == c).sum()) for c in (0, 1, 2)},
           "fractions": [], "mean_conf_before": float(p0.mean())}
    for f in fracs:
        w = int(round(f * n))
        cam_p, cam_pred = conf_after(best_window(cam, w, "top"), w)
        low_p, low_pred = conf_after(best_window(cam, w, "low"), w)
        rnd_drops, rnd_flips = [], []
        for _ in range(n_random):
            rs = rng.integers(0, n - w + 1, size=len(Y))
            rp, rpred = conf_after(rs, w)
            rnd_drops.append(p0 - rp)
            rnd_flips.append(rpred != Y)
        rnd_drop = np.mean(rnd_drops, axis=0)            # per-beat mean over random placements
        cam_drop, low_drop = p0 - cam_p, p0 - low_p
        row = {
            "fraction": f, "width_samples": w, "width_ms": round(w / FS * 1000),
            "drop_gradcam": float(cam_drop.mean()), "drop_random": float(rnd_drop.mean()),
            "drop_least_salient": float(low_drop.mean()),
            "flip_gradcam": float((cam_pred != Y).mean()), "flip_random": float(np.mean(rnd_flips)),
            "flip_least_salient": float((low_pred != Y).mean()),
            "share_gradcam_beats_random": float((cam_drop > rnd_drop).mean()),
            "per_class": {},
        }
        for c in (0, 1, 2):
            m = Y == c
            row["per_class"][CLASS_NAMES[c]] = {
                "drop_gradcam": float(cam_drop[m].mean()), "drop_random": float(rnd_drop[m].mean()),
                "drop_least_salient": float(low_drop[m].mean()),
                "flip_gradcam": float((cam_pred[m] != Y[m]).mean()),
                "flip_random": float(np.mean([f_[m].mean() for f_ in rnd_flips])),
            }
        out["fractions"].append(row)
    # paired bootstrap CI on the headline fraction (20%)
    f20 = next(r for r in out["fractions"] if abs(r["fraction"] - 0.2) < 1e-9)
    w = f20["width_samples"]
    cp, _ = conf_after(best_window(cam, w, "top"), w)
    rp, _ = conf_after(rng.integers(0, n - w + 1, size=len(Y)), w)
    diff = (p0 - cp) - (p0 - rp)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
    out["headline_20pct"] = {"mean_extra_drop_vs_random": float(diff.mean()),
                             "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    # a few example beats for the demo plot (one per class, highest-confidence correct beat among the sampled)
    ex = {}
    for c in (0, 1, 2):
        cand = np.where(Y == c)[0]
        j = cand[np.argsort(-p0[cand])[len(cand) // 4]]          # a typical, not the most extreme, beat
        w = int(round(0.2 * n))
        s = int(best_window(cam[j:j + 1], w, "top")[0])
        Xm = X[j:j + 1].copy(); Xm[0, s:s + w, 0] = 0.0
        pm = model.predict({"ecg_signal": Xm, "rr_features": R[j:j + 1]}, verbose=0)[0]
        ex[CLASS_NAMES[c]] = {"signal": X[j, :, 0].round(4).tolist(), "cam": cam[j].round(4).tolist(),
                              "start": s, "width": w, "conf_before": float(p0[j]),
                              "conf_after": float(pm[c]), "pred_after": CLASS_NAMES[int(pm.argmax())]}
    out["examples"] = ex
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--clf", default="arrhythmia_rr_img.keras")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(a.models, a.clf), compile=False)
    Xn, rr, y, rec = load_test(a.models)
    print(f"test beats: {len(y)}  records: {len(set(rec))}  class counts N/S/V/F: {np.bincount(y, minlength=4).tolist()}")
    prob = model.predict({"ecg_signal": Xn, "rr_features": rr}, batch_size=1024, verbose=0)
    np.savez_compressed(os.path.join(a.models, "eval_test_probs.npz"),
                        prob=prob.astype(np.float16), y=y.astype(np.int8), record=rec)
    print("saved eval_test_probs.npz   accuracy at argmax:", float((prob.argmax(1) == y).mean()))
    rng = np.random.default_rng(a.seed)
    res = faithfulness(model, Xn, rr, y, rng)
    res["data_note"] = ("REAL MIT-BIH beats (test patients), image-domain waveforms. Only beats the model "
                        "classified correctly are used. Masked region is set to the beat's mean level.")
    json.dump(res, open(os.path.join(a.models, "xai_faithfulness.json"), "w"), indent=1)
    for r in res["fractions"]:
        print(f"mask {r['fraction']:.0%}: drop Grad-CAM {r['drop_gradcam']:.3f} | random {r['drop_random']:.3f} "
              f"| least-salient {r['drop_least_salient']:.3f} | flips {r['flip_gradcam']:.1%} vs {r['flip_random']:.1%}")
    print("headline:", res["headline_20pct"])


if __name__ == "__main__":
    main()
