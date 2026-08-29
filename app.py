"""
app.py
======
Streamlit interface for the PVC Detection pipeline (CV + ML + XAI).

Sections:
  1. Single-Beat Detection  -- upload one isolated ECG beat image, get
     PVC/Normal prediction + Grad-CAM highlight + clinical explanation +
     the full Grad-CAM saliency curve (not just the collapsed highlight
     window -- shows exactly how much attention the CNN paid to every
     point along the beat).
  2. Multi-Beat Strip Detection -- upload a continuous multi-beat ECG
     strip image, get per-beat detection + one annotated strip image.
  3. Model Performance -- real training metrics (precision/recall/F1,
     confusion matrix, class distribution), read from
     models/train_metrics.json (written automatically by train.py).
  4. Saved Reports -- browse everything you've saved from tabs 1 and 2.

Run with:
    streamlit run app.py
(run this from the project root, i.e. the folder containing src/, models/,
data/, outputs/ -- same place you run `py -3.12 src/predict_pipeline.py` from)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.insert(0, "src")

import glob
import json
import tempfile
import textwrap
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from predict_pipeline import run_pipeline
from multi_beat_pipeline import run_multi_beat_pipeline

MODEL_PATH = "models/pvc_cnn.keras"
METRICS_PATH = "models/train_metrics.json"
SAVED_DIR = "saved_reports"
os.makedirs(SAVED_DIR, exist_ok=True)


# ---------------- Report-image composition ----------------

def _sanitize_text(s):
    """Replace Unicode punctuation PIL's default bitmap font can't render
    (shows up as broken box glyphs otherwise) with plain ASCII equivalents."""
    replacements = {
        "\u2014": "-", "\u2013": "-",       # em dash, en dash
        "\u2018": "'", "\u2019": "'",       # curly single quotes
        "\u201c": '"', "\u201d": '"',       # curly double quotes
        "\u2026": "...",                    # ellipsis
    }
    for uni, ascii_eq in replacements.items():
        s = s.replace(uni, ascii_eq)
    return s


def build_report_image(ecg_path, header_lines, body_lines, out_path):
    """Compose the ECG output image and its explanation text into ONE
    image, so a single saved/downloaded file always carries both together."""
    ecg_img = Image.open(ecg_path).convert("RGB")
    w, h = ecg_img.size

    try:
        font_header = ImageFont.load_default(size=16)
        font_body = ImageFont.load_default(size=13)
    except TypeError:
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()

    padding = 18
    line_height_header = 22
    line_height_body = 18
    max_chars = max(45, w // 8)

    header_lines = [_sanitize_text(l) for l in header_lines]
    body_lines = [_sanitize_text(l) for l in body_lines]

    wrapped_body = []
    for line in body_lines:
        wrapped_body.extend(textwrap.wrap(line, width=max_chars) or [""])

    text_height = (
        padding * 2
        + line_height_header * len(header_lines)
        + line_height_body * len(wrapped_body)
        + 10
    )
    canvas = Image.new("RGB", (w, h + text_height), "white")
    canvas.paste(ecg_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = h + padding
    for line in header_lines:
        draw.text((padding, y), line, fill="black", font=font_header)
        y += line_height_header
    y += 8
    for line in wrapped_body:
        draw.text((padding, y), line, fill="black", font=font_body)
        y += line_height_body

    canvas.save(out_path)
    return out_path


def save_to_history(report_path, kind, summary):
    """Copy a composed report image into SAVED_DIR with a timestamped,
    descriptive filename so the Saved Reports tab can browse it later."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_summary = "".join(c if c.isalnum() else "_" for c in summary)[:40]
    dest = os.path.join(SAVED_DIR, f"{ts}_{kind}_{safe_summary}.png")
    Image.open(report_path).save(dest)
    return dest


def plot_saliency(signal, saliency, highlight_window, is_pvc, out_path):
    """
    Two-panel plot: the ECG trace on top (with the PVC highlight band, if
    applicable), and the FULL Grad-CAM saliency curve underneath, sharing
    the same x-axis -- shows exactly how much attention the CNN paid to
    every point along the beat, not just the single collapsed highlight
    window used in the main output image.
    """
    signal = np.asarray(signal)
    saliency = np.asarray(saliency)
    x = np.arange(len(signal))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(x, signal, color="black", linewidth=1.2)
    ax1.set_ylabel("ECG signal")
    if is_pvc:
        hs, he = highlight_window
        ax1.axvspan(hs, he, color="red", alpha=0.2)

    ax2.fill_between(x, saliency, color="orange", alpha=0.6)
    ax2.plot(x, saliency, color="darkorange", linewidth=1)
    ax2.set_ylabel("Grad-CAM\nimportance")
    ax2.set_xlabel("Sample index")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# ---------------- Page setup ----------------

st.set_page_config(page_title="PVC Detection", layout="wide")
st.title("ECG PVC Detection")
st.caption(
    "Upload an ECG image to detect Premature Ventricular Contractions "
    "(PVC), with Grad-CAM highlighting and rule-based clinical explanation."
)

if not os.path.exists(MODEL_PATH):
    st.warning(
        f"⚠️ Model not found at `{MODEL_PATH}`. Train it first "
        f"(see README Step 2: `python src/train.py --mode real ...`), "
        f"or point MODEL_PATH in app.py at your trained model."
    )

tab_single, tab_multi, tab_perf, tab_saved = st.tabs(
    ["🫀 Single Beat", "📈 Multi-Beat Strip", "📊 Model Performance", "📁 Saved Reports"]
)


# ---------------- Single Beat Tab ----------------
with tab_single:
    st.subheader("Single-Beat Detection")
    st.write("Upload an image containing **one isolated ECG beat**.")

    single_file = st.file_uploader(
        "Upload single-beat ECG image", type=["png", "jpg", "jpeg"], key="single"
    )

    if single_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Input Image**")
            st.image(single_file, width=480)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(single_file.getvalue())
            tmp_path = tmp.name

        result = None
        if os.path.exists(MODEL_PATH):
            with st.spinner("Running CV digitization → CNN → XAI..."):
                try:
                    result = run_pipeline(tmp_path, MODEL_PATH, out_dir="outputs")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

        if result is not None:
            header = [
                f"Prediction: {result['prediction']}  "
                f"(p={result['pvc_probability']:.3f})"
            ]
            report_path = os.path.join(
                "outputs",
                os.path.splitext(os.path.basename(result["output_image"]))[0]
                + "_report.png",
            )
            build_report_image(
                result["output_image"], header, result["reasons"], report_path
            )

            with col2:
                st.write("**Explained Output (image + explanation)**")
                st.image(report_path, width=480)

                dl_col, save_col = st.columns(2)
                with dl_col:
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download report",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="image/png",
                            key="download_single",
                        )
                with save_col:
                    if st.button("💾 Save to history", key="save_single"):
                        dest = save_to_history(
                            report_path, "single", result["prediction"]
                        )
                        st.success(f"Saved to {dest}")

            if result["prediction"] == "PVC":
                st.error(
                    f"**Prediction: PVC**  "
                    f"(probability = {result['pvc_probability']:.3f})"
                )
            else:
                st.success(
                    f"**Prediction: Normal**  "
                    f"(probability = {result['pvc_probability']:.3f})"
                )

            st.write("**Clinical Explanation:**")
            for r in result["reasons"]:
                st.write(f"- {r}")

            # Full Grad-CAM saliency curve, not just the collapsed
            # highlight window -- shows the CNN's attention across the
            # ENTIRE beat, point by point.
            if "signal" in result and "saliency_curve" in result:
                st.write("**Grad-CAM Attention Curve**")
                st.caption(
                    "How much the CNN 'looked at' each point of the beat "
                    "when making its decision -- the highlight band in the "
                    "image above is just this curve's peak region."
                )
                saliency_path = os.path.join(
                    "outputs",
                    os.path.splitext(os.path.basename(result["output_image"]))[0]
                    + "_saliency.png",
                )
                plot_saliency(
                    result["signal"],
                    result["saliency_curve"],
                    result["highlight_window_samples"],
                    result["prediction"] == "PVC",
                    saliency_path,
                )
                st.image(saliency_path, width=550)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------- Multi-Beat Tab ----------------
with tab_multi:
    st.subheader("Multi-Beat Strip Detection")
    st.write(
        "Upload an image containing a **continuous strip with multiple "
        "consecutive beats** (e.g. a Lead II rhythm strip)."
    )

    multi_file = st.file_uploader(
        "Upload multi-beat ECG strip image", type=["png", "jpg", "jpeg"], key="multi"
    )

    if multi_file is not None:
        st.write("**Input Image**")
        st.image(multi_file, width=700)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(multi_file.getvalue())
            tmp_path = tmp.name

        result = None
        if os.path.exists(MODEL_PATH):
            with st.spinner("Detecting beats → CNN → XAI on each beat..."):
                try:
                    result = run_multi_beat_pipeline(
                        tmp_path, MODEL_PATH, out_dir="outputs"
                    )
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

        if result is not None:
            header = [
                f"{result['num_beats']} beats analyzed | "
                f"{result['num_pvc']} flagged as PVC"
            ]
            body = []
            for b in result["beats"]:
                marker = "PVC" if b["prediction"] == "PVC" else "Normal"
                body.append(
                    f"Beat {b['beat_index'] + 1}: {marker} "
                    f"(p={b['pvc_probability']:.3f})"
                )
                for r in b["reasons"]:
                    body.append(f"    - {r}")

            report_path = os.path.join(
                "outputs",
                os.path.splitext(os.path.basename(result["output_image"]))[0]
                + "_report.png",
            )
            build_report_image(result["output_image"], header, body, report_path)

            st.write("**Annotated Output + Explanation**")
            st.image(report_path, width=700)

            dl_col, save_col = st.columns(2)
            with dl_col:
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download report",
                        data=f.read(),
                        file_name=os.path.basename(report_path),
                        mime="image/png",
                        key="download_multi",
                    )
            with save_col:
                if st.button("💾 Save to history", key="save_multi"):
                    dest = save_to_history(
                        report_path,
                        "multibeat",
                        f"{result['num_beats']}beats_{result['num_pvc']}pvc",
                    )
                    st.success(f"Saved to {dest}")

            st.write(
                f"### {result['num_beats']} beats analyzed — "
                f"{result['num_pvc']} flagged as PVC"
            )

            for b in result["beats"]:
                is_pvc = b["prediction"] == "PVC"
                label = "🔴 PVC" if is_pvc else "🟢 Normal"
                with st.expander(
                    f"Beat {b['beat_index'] + 1}: {label}  "
                    f"(p={b['pvc_probability']:.3f})"
                ):
                    if b["reasons"]:
                        for r in b["reasons"]:
                            st.write(f"- {r}")
                    else:
                        st.write("Beat morphology falls within normal limits.")

                    if "signal" in b and "saliency_curve" in b:
                        beat_saliency_path = os.path.join(
                            "outputs", f"multibeat_saliency_beat{b['beat_index']+1}.png"
                        )
                        plot_saliency(
                            b["signal"],
                            b["saliency_curve"],
                            b["local_highlight_window"],
                            is_pvc,
                            beat_saliency_path,
                        )
                        st.image(beat_saliency_path, width=550)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------- Model Performance Tab ----------------
with tab_perf:
    st.subheader("Model Performance")

    if not os.path.exists(METRICS_PATH):
        st.info(
            f"No metrics file found at `{METRICS_PATH}`. This is written "
            f"automatically the next time you run `train.py`. If you "
            f"already have real training numbers, save them there in the "
            f"same JSON shape train.py produces."
        )
    else:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        st.write(
            f"Trained on **{metrics['dataset']['total_beats']:,} beats** "
            f"from MIT-BIH records: `{', '.join(metrics.get('records') or ['demo'])}`"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Distribution**")
            st.bar_chart(
                {
                    "Normal": metrics["dataset"]["normal_count"],
                    "PVC": metrics["dataset"]["pvc_count"],
                }
            )

        with col2:
            st.write("**Confusion Matrix**")
            cm = np.array(metrics["confusion_matrix"])
            labels = metrics["confusion_matrix_labels"]
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
            ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            plt.tight_layout()
            st.pyplot(fig)

        st.write("**Classification Report**")
        rep = metrics["classification_report"]
        rows = []
        for cls in ["Normal", "PVC"]:
            rows.append({
                "Class": cls,
                "Precision": f"{rep[cls]['precision']:.2f}",
                "Recall": f"{rep[cls]['recall']:.2f}",
                "F1-score": f"{rep[cls]['f1-score']:.2f}",
                "Support": int(rep[cls]["support"]),
            })
        st.table(rows)

        pvc_recall = rep["PVC"]["recall"]
        st.info(
            f"**PVC recall = {pvc_recall:.0%}** — for a medical detection "
            f"task, recall on the positive class matters more than raw "
            f"accuracy, since missing a true PVC is worse than a false "
            f"alarm. Out of {int(rep['PVC']['support'])} real PVC beats "
            f"in the test set, this model correctly caught "
            f"{round(pvc_recall * rep['PVC']['support'])}."
        )


# ---------------- Saved Reports Tab ----------------
with tab_saved:
    st.subheader("Saved Reports")
    st.write(
        "Everything you've saved with **💾 Save to history** from the "
        "Single Beat and Multi-Beat tabs shows up here, newest first."
    )

    saved_files = sorted(
        glob.glob(os.path.join(SAVED_DIR, "*.png")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not saved_files:
        st.info(
            "No saved reports yet. Run a detection in the Single Beat or "
            "Multi-Beat tab, then click **💾 Save to history**."
        )
    else:
        st.write(f"**{len(saved_files)} saved report(s)**")
        for path in saved_files:
            fname = os.path.basename(path)
            with st.expander(fname):
                st.image(path, width=550)
                col_dl, col_del = st.columns(2)
                with col_dl:
                    with open(path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f.read(),
                            file_name=fname,
                            mime="image/png",
                            key=f"dl_{fname}",
                        )
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{fname}"):
                        os.remove(path)
                        st.rerun()