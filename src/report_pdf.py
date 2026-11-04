"""
report_pdf.py
==============
One-page-plus clinical-style PDF report for an analysed ECG strip (ReportLab + matplotlib).

    from report_pdf import build_pdf
    build_pdf(result, "outputs/report.pdf", explain_fn=lambda i: explain_beat(result, i))

Contents: annotated strip, findings summary, rhythm statistics, beat table, evidence for flagged beats
(Grad-CAM + rule-based reasons), limitations. It describes patterns of detected beats; it is NOT a diagnosis.
"""
import os
import io
from datetime import datetime
import numpy as np

CLASS_FULL = {"N": "Normal", "S": "Supraventricular (PAC-type)", "V": "Ventricular (PVC)", "U": "Uncertain"}


def _beat_figure(b, saliency, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = np.asarray(b["window"]).ravel()
    t = (np.arange(len(w)) - 90) / 360.0 * 1000
    fig, ax = plt.subplots(figsize=(3.3, 1.7), dpi=140)
    ax.plot(t, w, color="black", lw=1.1)
    if saliency is not None:
        ax.fill_between(t, w.min(), w.max(), where=saliency > 0.5, color="#d62828", alpha=0.25)
    ax.set_xlabel("ms from R peak", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_pdf(result, out_path, explain_fn=None, title="ECG Arrhythmia Analysis Report", max_detail=6):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                    KeepTogether, CondPageBreak)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8.5, leading=11)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=13)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title=title)
    S = []
    S.append(Paragraph(title, ss["Title"]))
    S.append(Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; Source image: "
                       f"{os.path.basename(result['image_path'])} &nbsp;|&nbsp; "
                       f"Analysis mode: {result['mode']}", small))
    S.append(Spacer(1, 4))

    ov = result.get("overlay_path")
    if ov and os.path.exists(ov):
        from PIL import Image as PI
        iw, ih = PI.open(ov).size
        wmm = 178
        S.append(Image(ov, width=wmm * mm, height=wmm * mm * ih / iw))

    S.append(Paragraph("Findings", h2))
    S.append(Paragraph(result["summary"], body))
    for n in result.get("notes", []):
        S.append(Paragraph(f"<i>Note: {n}</i>", small))

    r = result.get("rhythm")
    if r:
        hr = r.get("heart_rate") or {}
        rows = [["Measure", "Value"],
                ["Beats analysed", str(r["n_beats"])],
                ["Duration", f"{r['duration_s']} s"],
                ["Heart rate (mean)", f"{hr.get('mean_bpm', '-')} bpm"],
                ["Heart rate (5-beat range)", f"{hr.get('min_bpm', '-')} - {hr.get('max_bpm', '-')} bpm"],
                ["Normal / Supraventricular / Ventricular beats",
                 f"{r['counts']['N']} / {r['counts']['S']} / {r['counts']['V']}"],
                ["PVC patterns", f"{r['V']['isolated']} isolated, {r['V']['couplets']} couplet(s), "
                                 f"{len(r['V']['runs_ge3'])} run(s) of 3+, {r['V']['bigeminy_episodes']} bigeminy, "
                                 f"{r['V']['trigeminy_episodes']} trigeminy"],
                ["Beats below confidence threshold", f"{r.get('n_uncertain', 0)} (threshold {result.get('conf_thr', 0):.0%})"]]
        t = Table(rows, colWidths=[78 * mm, 98 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a44")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8bfcc")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f9")])]))
        S.append(Paragraph("Rhythm statistics", h2))
        S.append(t)

    S.append(Paragraph("Beat-by-beat", h2))
    rows = [["#", "Class", "Confidence", "R-R before", "Evidence (short)"]]
    for b in result["beats"]:
        ev = b["reasons"][1] if len(b.get("reasons", [])) > 1 else (b["reasons"][0] if b.get("reasons") else "")
        if len(ev) > 120:
            ev = ev[:120].rsplit(" ", 1)[0] + " ..."
        rows.append([str(b["index"] + 1), CLASS_FULL.get(b["label"], b["label"]),
                     f"{b['confidence']:.0%}" if "confidence" in b else "-",
                     f"{b['rr_prev_s']:.2f} s" if "rr_prev_s" in b else "-",
                     Paragraph(ev, small)])
    t = Table(rows, colWidths=[9 * mm, 42 * mm, 21 * mm, 21 * mm, 83 * mm], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a44")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8bfcc")),
             ("VALIGN", (0, 0), (-1, -1), "TOP")]
    pal = {"V": "#fde2e2", "S": "#fff0d6", "U": "#ececec"}
    for i, b in enumerate(result["beats"], start=1):
        if b["label"] in pal:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(pal[b["label"]])))
    t.setStyle(TableStyle(style))
    S.append(t)

    flagged = [b for b in result["beats"] if b["label"] in ("V", "S", "U")][:max_detail]
    if flagged and "window" in flagged[0]:
        S.append(CondPageBreak(70 * mm))
        S.append(Paragraph("Evidence for flagged beats (Grad-CAM = red shading on the beat; rule-based reasons)", h2))
        tmp = os.path.splitext(out_path)[0] + "_tmp"
        os.makedirs(tmp, exist_ok=True)
        for b in flagged:
            sal = None
            if explain_fn is not None and result["mode"] == "calibrated":
                try:
                    sal = explain_fn(b["index"])
                except Exception:
                    sal = None
            elif "saliency" in b:
                sal = np.asarray(b["saliency"])
            fp = os.path.join(tmp, f"beat{b['index']}.png")
            _beat_figure(b, sal, fp)
            txt = "<br/>".join(f"&bull; {x}" for x in b["reasons"])
            cell = Table([[Image(fp, width=58 * mm, height=30 * mm),
                           Paragraph(f"<b>Beat {b['index'] + 1}: {CLASS_FULL.get(b['label'], b['label'])}</b><br/>{txt}",
                                     small)]], colWidths=[62 * mm, 114 * mm])
            cell.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8bfcc"))]))
            S.append(KeepTogether([cell, Spacer(1, 3)]))

    S.append(Paragraph("Limitations", h2))
    S.append(Paragraph(
        "This report describes patterns found in an ECG image by an automated system trained on the public MIT-BIH "
        "Arrhythmia Database (evaluated patient-wise on unseen records). Beats read from an image are less reliable than "
        "from a recorded signal; supraventricular (PAC-type) detection is the weakest class, and class F (fusion) is not "
        "reported. Beats under the confidence threshold are marked Uncertain instead of being forced into a class. "
        "<b>This is a student research prototype, not a medical device, and not a diagnosis.</b>", small))
    doc.build(S)
    try:
        import shutil
        shutil.rmtree(os.path.splitext(out_path)[0] + "_tmp", ignore_errors=True)
    except Exception:
        pass
    return out_path
