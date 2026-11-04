"""
app_pro.py  --  ECG Arrhythmia Analysis (multi-class) with Explainable AI
==========================================================================
Run from the project root (same folder as app.py, src/, models/, data/):

    streamlit run app_pro.py

Tabs
  1. Analyze ECG      upload an ECG image, pick a gallery strip, or pick one of the existing demo images.
                      Output: beats coloured N / S / V (or Uncertain), heart rate, PVC count and patterns,
                      beat inspector with Grad-CAM + rule-based reasons, downloadable PDF report.
  2. Rhythm & HRV     R-R intervals of the analysed strip: pNN50, Poincare plot (SD1/SD2), R-R distribution, ectopic timing.
  3. How it works     the computer-vision stage made visible (U-Net trace, grid calibration, R peaks) + pipeline diagram.
  4. Performance      real, labelled numbers (real signals vs rendered images) and the limits.
  5. Evaluation lab   interactive threshold trade-off + per-patient results on REAL MIT-BIH test patients.
  6. XAI faithfulness Grad-CAM deletion test: does hiding the highlighted region really hurt the prediction?
  7. Single beat      the original single-beat PVC / Normal detector with Grad-CAM (unchanged logic from app.py).

Run once before the new dashboards (creates models/eval_test_probs.npz and models/xai_faithfulness.json):
    py -3.12 src/dashboard_data.py

The original app.py is NOT modified; this is a separate, additional app.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import glob
import json
import tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "cv_unet")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "outputs")
GALLERY = os.path.join(ROOT, "data", "ecg_images", "gallery")
DEMOS = os.path.join(ROOT, "data", "ecg_images")
os.makedirs(OUT, exist_ok=True)

from arrhythmia_pipeline import analyze_strip, relabel, explain_beat, COLORS, CLASS_FULL

st.set_page_config(page_title="ECG Arrhythmia Analysis", layout="wide")


# ----------------------------------------------------------------------------- helpers
def show_img(img, caption=None):
    """Full-width image; works across Streamlit versions (width='stretch' is new, use_container_width is old)."""
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(img, caption=caption, **kw)
            return
        except Exception:
            continue


def chips(beats):
    html = []
    for b in beats:
        lab = b["label"]
        txt = "?" if lab == "U" else lab
        html.append(f"<span style='display:inline-block;margin:2px;padding:3px 9px;border-radius:12px;"
                    f"background:{COLORS[lab]};color:white;font-weight:600;font-family:monospace'>{txt}</span>")
    return " ".join(html)


def beat_figure(b, saliency=None):
    w = np.asarray(b["window"]).ravel()
    t = (np.arange(len(w)) - 90) / 360.0 * 1000
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=110)
    ax.plot(t, w, color="black", lw=1.4)
    if saliency is not None:
        sal = np.asarray(saliency)
        ax.fill_between(t, w.min() - 0.05, w.max() + 0.05, where=sal > 0.5, color="#d62828", alpha=0.22,
                        label="Grad-CAM: where the CNN looked")
        ax2 = ax.twinx()
        ax2.fill_between(t, sal, color="#f08c00", alpha=0.25)
        ax2.set_ylim(0, 3)
        ax2.set_yticks([])
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.axvline(0, color="#999", lw=0.6, ls=":")
    ax.set_xlabel("ms from R peak")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def trace_overlay(result):
    bgr = cv2.imread(result["image_path"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    m = result["mask"] > 0
    ov = rgb.copy()
    ov[m] = (255, 0, 0)
    return ov


def resolve_source():
    src = st.radio("Image source", ["Gallery (unseen patients)", "Upload an image", "Existing demo images"],
                   horizontal=True)
    truth = None
    if src.startswith("Gallery"):
        gj = os.path.join(GALLERY, "gallery.json")
        if not os.path.exists(gj):
            st.info("Gallery not built yet. Run:  `py -3.12 src/make_demo_gallery.py --data_dir data/mitdb "
                    "--models_dir models --out_dir data/ecg_images/gallery`")
            return None, None
        gal = json.load(open(gj))
        pick = st.selectbox("Example", range(len(gal)),
                            format_func=lambda i: f"{gal[i]['title']}  (record {gal[i]['record']})")
        e = gal[pick]
        return os.path.join(GALLERY, e["file"]), e
    if src.startswith("Upload"):
        f = st.file_uploader("ECG image (PNG / JPG)", type=["png", "jpg", "jpeg"])
        if f is None:
            return None, None
        d = os.path.join(tempfile.gettempdir(), "ecg_uploads")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f.name)
        open(p, "wb").write(f.getvalue())
        return p, None
    files = sorted(glob.glob(os.path.join(DEMOS, "demo_strip_*.png")) + glob.glob(os.path.join(DEMOS, "test_strip_*.png"))
                   + glob.glob(os.path.join(DEMOS, "user_test_strip.png")))
    if not files:
        st.info("No demo images found in data/ecg_images.")
        return None, None
    p = st.selectbox("Demo image", files, format_func=os.path.basename)
    return p, None


def compare_with_truth(result, entry):
    dur = result["duration_s"]
    pred = [(b["peak"] / 360.0, b["label"]) for b in result["beats"]]
    rows, ok_n = [], 0
    for t in entry["truth"]:
        best = min(pred, key=lambda p: abs(p[0] - t["t"])) if pred else None
        if best and abs(best[0] - t["t"]) <= 0.15:
            good = best[1] == t["label"]
            ok_n += good
            rows.append({"time (s)": round(t["t"], 2), "true (MIT-BIH annotation)": t["label"],
                         "predicted": "uncertain" if best[1] == "U" else best[1], "match": "✅" if good else "❌"})
        else:
            rows.append({"time (s)": round(t["t"], 2), "true (MIT-BIH annotation)": t["label"],
                         "predicted": "not detected", "match": "❌"})
    return rows, ok_n


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Confidence threshold", 0.40, 0.90, 0.55, 0.05,
                    help="A beat whose top-class confidence is below this is shown as 'Uncertain' "
                         "instead of being forced into a class. Raise it for fewer mistakes, "
                         "lower it for fewer uncertain beats.")
    st.markdown("**Legend**")
    for k in ("N", "S", "V", "U"):
        st.markdown(f"<span style='color:{COLORS[k]}'>■</span> {k if k != 'U' else '?'} &nbsp; {CLASS_FULL[k]}",
                    unsafe_allow_html=True)
    st.caption("Research prototype, not a medical device.")

st.title("ECG Arrhythmia Analysis")

if not os.path.exists(os.path.join(MODELS, "arrhythmia_rr_img.keras")):
    st.error("models/arrhythmia_rr_img.keras not found. Train it first (see the Performance tab for the commands).")

tab_an, tab_rhythm, tab_how, tab_perf, tab_eval, tab_xai, tab_single = st.tabs(
    ["Analyze ECG", "Rhythm & HRV", "Working", "Performance", "Evaluation lab",
     "XAI faithfulness", "Single beat"])

# ----------------------------------------------------------------------------- tab 1
with tab_an:
    path, entry = resolve_source()
    res = None
    if path:
        key = (path, os.path.getmtime(path))
        cache = st.session_state.setdefault("cache", {})
        if key not in cache:
            with st.spinner("Reading image → tracing waveform → detecting beats → classifying…"):
                try:
                    cache[key] = analyze_strip(path, MODELS, OUT, conf_thr=thr)
                except Exception as e:
                    cache[key] = e
        res = cache[key]
        if isinstance(res, Exception):
            st.error(f"Analysis failed: {type(res).__name__}: {res}")
            st.caption("Tips: crop tightly around the strip, use a clean screenshot, avoid phone glare.")
            res = None
        else:
            res = relabel(res, thr, OUT) if res["mode"] == "calibrated" else res

    if path and res:
        if res["mode"] == "calibrated":
            st.success(f"ECG grid detected ({res['calibration']['px_per_mm']:.1f} px/mm, "
                       f"{res['duration_s']:.1f} s shown) → full analysis with real time scale.")
        else:
            st.warning("No usable ECG paper grid found → **shape-only mode** (Normal vs PVC per beat; "
                       "no heart rate or rhythm patterns because time is unknown).")
        for n in res.get("notes", []):
            if "shape-only" not in n.lower() and "No usable" not in n:
                st.caption(f"ℹ️ {n}")

        c1, c2 = st.columns([1.15, 1])
        with c1:
            show_img(res["overlay_path"], "Annotated strip (N green · S orange · V red · ? grey)")
        with c2:
            r = res.get("rhythm")
            if r:
                hr = (r.get("heart_rate") or {}).get("mean_bpm")
                m1, m2, m3 = st.columns(3)
                m1.metric("Beats", r["n_beats"])
                m2.metric("Heart rate", f"{hr:.0f} bpm" if hr else "-")
                m3.metric("Uncertain", r.get("n_uncertain", 0))
                m4, m5, _ = st.columns(3)
                m4.metric("PVC (V)", r["counts"]["V"])
                m5.metric("PAC-type (S)", r["counts"]["S"])
            else:
                st.metric("Beats analysed", res["n_beats"])
                st.metric("Flagged as PVC", res.get("n_pvc", 0))
            st.markdown("**Beat sequence**")
            st.markdown(chips(res["beats"]), unsafe_allow_html=True)
            st.markdown("**Summary**")
            st.write(res["summary"])
            if r and res["mode"] == "calibrated":
                V = r["V"]
                pats = []
                if V["couplets"]:
                    pats.append(f"{V['couplets']} couplet(s)")
                if V["runs_ge3"]:
                    pats.append(f"{len(V['runs_ge3'])} run(s) of ≥3 PVCs")
                if V["bigeminy_episodes"]:
                    pats.append("bigeminy")
                if V["trigeminy_episodes"]:
                    pats.append("trigeminy")
                if pats:
                    st.info("PVC pattern: " + ", ".join(pats))

        if entry:
            rows, ok_n = compare_with_truth(res, entry)
            with st.expander(f"Ground truth vs prediction — {ok_n}/{len(rows)} beats match (real MIT-BIH annotations)",
                             expanded=False):
                st.dataframe(rows, hide_index=True)
                st.caption("Source: real MIT-BIH annotations, record "
                           f"{entry['record']} (a test patient, never used for training). Image is a clean synthetic render.")

        st.divider()
        st.subheader("Beat inspector")
        if not res["beats"]:
            st.info("No beats to inspect.")
        else:
            labels = [f"Beat {b['index'] + 1} — {CLASS_FULL[b['label']] if b['label'] != 'U' else 'Uncertain'}"
                      f" ({b['confidence']:.0%})" for b in res["beats"]]
            default = next((i for i, b in enumerate(res["beats"]) if b["label"] in ("V", "S")), 0)
            bi = st.selectbox("Choose a beat", range(len(labels)), index=default, format_func=lambda i: labels[i])
            b = res["beats"][bi]
            ca, cb = st.columns([1.1, 1])
            with ca:
                sal = None
                if res["mode"] == "calibrated":
                    sk = (key, bi)
                    sc = st.session_state.setdefault("sal", {})
                    if sk not in sc:
                        try:
                            sc[sk] = explain_beat(res, bi, MODELS)
                        except Exception:
                            sc[sk] = None
                    sal = sc[sk]
                else:
                    sal = np.asarray(b.get("saliency")) if b.get("saliency") is not None else None
                fig = beat_figure(b, sal)
                st.pyplot(fig)
                plt.close(fig)
            with cb:
                if res["mode"] == "calibrated":
                    st.markdown("**Class probabilities** (F is ignored)")
                    st.bar_chart({k: v for k, v in b["probs"].items()}, horizontal=True)
                st.markdown("**Why this call** (rule-based clinical reasoning)")
                for rsn in b["reasons"]:
                    st.write(f"- {rsn}")

        st.divider()
        try:
            from report_pdf import build_pdf
            pdf_path = os.path.join(OUT, os.path.splitext(os.path.basename(path))[0] + "_report.pdf")
            if st.button("📄 Generate PDF report"):
                with st.spinner("Building report…"):
                    exp = (lambda i: explain_beat(res, i, MODELS)) if res["mode"] == "calibrated" else None
                    build_pdf(res, pdf_path, explain_fn=exp)
                st.session_state["pdf"] = pdf_path
            if st.session_state.get("pdf") and os.path.exists(st.session_state["pdf"]):
                with open(st.session_state["pdf"], "rb") as f:
                    st.download_button("⬇️ Download PDF report", f.read(),
                                       file_name=os.path.basename(st.session_state["pdf"]), mime="application/pdf")
        except ImportError:
            st.caption("PDF export needs ReportLab:  `py -3.12 -m pip install reportlab`")

# ----------------------------------------------------------------------------- extra dashboards
import dash_tabs
with tab_rhythm:
    dash_tabs.render_rhythm_hrv(res)
with tab_eval:
    dash_tabs.render_eval_lab(MODELS, thr)
with tab_xai:
    dash_tabs.render_xai_faithfulness(MODELS)

# ----------------------------------------------------------------------------- tab 2
with tab_how:
    st.subheader("Pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
      img [label="ECG image\\n(photo / screenshot / scan)"];
      crop [label="Paper auto-crop\\n+ grid calibration\\n(px per second)"];
      unet [label="U-Net\\ntrace segmentation"];
      sig [label="1-D signal\\n@ 360 Hz"];
      det [label="R-peak\\ndetection"];
      cnn [label="1D-CNN + R-R features\\nN / S / V + confidence"];
      xai [label="Explainable AI\\nGrad-CAM + clinical rules"];
      rhy [label="Rhythm analytics\\nHR, PVC count, couplets,\\nbigeminy, runs"];
      out [label="Annotated strip\\n+ PDF report", fillcolor="#e6f6ea"];
      img -> crop -> unet -> sig -> det -> cnn; cnn -> xai; cnn -> rhy; xai -> out; rhy -> out; }
    """)
    if path and res and res["mode"] == "calibrated":
        st.subheader("This image, step by step")
        a, b_ = st.columns(2)
        with a:
            show_img(trace_overlay(res), "U-Net trace (red)")
        with b_:
            fig, ax = plt.subplots(figsize=(7, 2.8), dpi=110)
            sig = res["signal"]
            tt = np.arange(len(sig)) / 360.0
            ax.plot(tt, sig, color="black", lw=1)
            for bt in res["beats"]:
                ax.plot(bt["peak"] / 360.0, sig[bt["peak"]], "o", color=COLORS[bt["label"]], ms=7)
            ax.set_xlabel("seconds")
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Digitized signal at 360 Hz with detected R peaks (coloured by class).")
        cal = res["calibration"]
        st.write(f"Grid calibration: **{cal['px_per_mm']:.2f} px/mm** → {cal['px_per_s']:.0f} px per second at 25 mm/s "
                 f"(confidence {cal['confidence']:.2f}; {cal['note']}).")
    else:
        st.caption("Analyze an image with a visible ECG grid in the first tab to see its CV steps here.")

# ----------------------------------------------------------------------------- tab 3
with tab_perf:
    bj = os.path.join(MODELS, "benchmark_results.json")
    if os.path.exists(bj):
        b = json.load(open(bj))
        st.markdown("**A. Benchmark on REAL MIT-BIH signals (clean signal, no image step)** — mean of "
                    f"{b.get('seeds', '?')} seeds")
        names = {"rf": "Random Forest", "svm": "SVM", "cnn": "1D-CNN (signal only)", "cnn_rr": "1D-CNN + R-R (ours)",
                 "resnet_rr": "ResNet + R-R", "bilstm_rr": "BiLSTM + R-R"}
        rows = []
        for k, v in b["results"].items():
            mean = lambda f: float(np.mean([r[f] for r in v]))
            rows.append({"Model": names.get(k, k), "Macro-F1": round(mean("macro_f1"), 3),
                         "Accuracy": round(mean("accuracy"), 3), "PVC sensitivity": round(mean("v_sens"), 3),
                         "PVC precision": round(mean("v_ppv"), 3), "PAC-type (S) F1": round(mean("s_f1"), 3)})
        st.dataframe(rows, hide_index=True)

    ij = os.path.join(MODELS, "arrhythmia_rr_img_metrics.json")
    if os.path.exists(ij):
        m = json.load(open(ij))
        rep = m["test"]["report"]
        st.markdown("**B. The classifier the app runs — REAL MIT-BIH beats, SYNTHETIC rendered images** "
                    f"(accuracy {m['test']['accuracy']:.3f}, macro-F1 {m['test']['macro_f1']:.3f}, 4 classes incl. F)")
        st.dataframe([{"Class": c, "Precision": round(rep[c]["precision"], 3), "Recall": round(rep[c]["recall"], 3),
                       "F1": round(rep[c]["f1-score"], 3), "Support": int(rep[c]["support"])}
                      for c in ("N", "S", "V", "F")], hide_index=True)
        cm = os.path.join(MODELS, "arrhythmia_rr_img_confusion.png")
        if os.path.exists(cm):
            show_img(cm, "Confusion matrix (test patients)")

    pj = os.path.join(GALLERY, "pipeline_eval_clean.json")
    if os.path.exists(pj):
        e = json.load(open(pj))
        st.markdown("**C. Whole pipeline on clean rendered images — REAL MIT-BIH beats from unseen patients, "
                    f"SYNTHETIC images** ({e['strips']} strips, {e['true_beats_scored']} beats)")
        st.dataframe([{"Class": c, "Support": v["support"], "Sensitivity": round(v["sensitivity"], 3),
                       "Precision": round(v["precision"], 3), "F1": round(v["f1"], 3)}
                      for c, v in e["per_class"].items()], hide_index=True)
        st.caption(f"Beat detection sensitivity {e['beat_detection_sensitivity']:.3f}; "
                   f"{e['false_detections']} false detections; "
                   f"{e['uncertain_share_of_detected']:.1%} of detected beats were 'uncertain'. "
                   "Strips are centred on S/V beats about half the time, so class mix is richer than natural prevalence.")

    st.markdown("**Known limits (stated openly)**")
    st.markdown(
        "- **PVC (V)** is the strongest class. **PAC-type (S)** is hard: the beat looks normal and its tiny P wave "
        "is blurred when a signal is read back from an image.\n"
        "- **Fusion (F)** cannot be learned reliably (almost all F beats come from one patient) and is not reported.\n"
        "- Results are on **rendered** images; real photos (glare, tilt) are harder and are not claimed here.")

# ----------------------------------------------------------------------------- tab 4
with tab_single:
    st.subheader("Single-beat PVC detector")
    st.write("Upload an image containing **one isolated beat** (original binary detector + Grad-CAM, unchanged).")
    f = st.file_uploader("Single-beat ECG image", type=["png", "jpg", "jpeg"], key="single_up")
    mp = os.path.join(MODELS, "pvc_cnn.keras")
    if f is not None and os.path.exists(mp):
        from predict_pipeline import run_pipeline
        tmp = os.path.join(tempfile.gettempdir(), "single_" + f.name)
        open(tmp, "wb").write(f.getvalue())
        with st.spinner("Running…"):
            try:
                r1 = run_pipeline(tmp, mp, out_dir=OUT)
            except Exception as e:
                r1 = None
                st.error(f"Pipeline failed: {e}")
        if r1:
            c1, c2 = st.columns(2)
            with c1:
                st.image(f.getvalue(), caption="Input", width=420)
            with c2:
                show_img(r1["output_image"], "Explained output")
            (st.error if r1["prediction"] == "PVC" else st.success)(
                f"Prediction: {r1['prediction']}  (probability {r1['pvc_probability']:.3f})")
            for rr_ in r1["reasons"]:
                st.write(f"- {rr_}")
