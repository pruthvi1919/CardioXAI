# CardioXAI: Deep Learning & Explainable AI for ECG Arrhythmia Classification

CardioXAI reads an **ECG image** (a scan, screenshot or photo of a rhythm strip), digitizes the trace
with computer vision, classifies every heartbeat with 1D CNNs trained on real MIT-BIH data, and
**explains each call** with Grad-CAM saliency plus rule-based clinical reasoning (P wave, QRS width,
T wave). It also reports rhythm statistics (heart rate, HRV, ectopic patterns) and exports a PDF report.

The project started as a PVC (Premature Ventricular Contraction) detector and has grown into a
multi-class arrhythmia analysis tool. The original PVC-only pipeline is still included (`app.py`).

> Research / educational project. **Not a medical device** and not for diagnosis.

---

## Pipeline

![Original PVC pipeline](pvc_pipeline_professional.png)

The current multi-class pipeline (`src/arrhythmia_pipeline.py`, used by `app_pro.py`):

```
ECG image
  -> paper auto-crop + grid calibration (pixels per second)       cv_unet/paper_crop.py, grid_calibration.py
  -> U-Net trace segmentation -> 1-D signal at 360 Hz             cv_unet/digitize_unet.py
  -> R-peak detection                                             beat_detect.py
  -> per-beat 1D CNN + R-R interval features -> N / S / V / Uncertain
  -> Grad-CAM (predicted class) + rule-based reasons              xai_explain.py
  -> rhythm analytics (heart rate, couplets, runs, bigeminy ...)  rhythm_analytics.py
  -> annotated image + PDF report                                 report_pdf.py
```

Beat classes follow the AAMI standard: **N** normal, **S** supraventricular ectopic (PAC-type),
**V** ventricular ectopic (PVC), **F** fusion. Beats below a confidence threshold are shown as
"Uncertain" instead of being forced into a class.

Two modes are chosen automatically:
- **Calibrated**: a real ECG paper grid is found, so time is known in seconds and R-R based
  statements (heart rate, rhythm patterns) are made.
- **Shape-only**: no usable grid (for example images from `ecg_image_gen.py`). Time is unknown, so
  R-R statements are skipped and each beat is classified Normal vs PVC by the original binary CNN.

---

## Results (real MIT-BIH data, test patients never used for training)

Evaluation is **inter-patient** (train on DS1 patients, test on DS2 patients), so no patient appears
in both sets. Beats and labels are real MIT-BIH annotations.

| Model | Input | PVC recall | Accuracy | Normal recall |
|---|---|---|---|---|
| CNN + R-R, 4-class | Raw MIT-BIH signals (49,690 test beats) | **95.9%** | **94.7%** | **97.8%** |
| CNN + R-R, 4-class (the model `app_pro.py` runs) | Rendered ECG images of real beats (6,724 test beats) | 83.0% | 78.4% | 84.2% |
| Original binary PVC detector | Raw signals, random beat-wise split* | 98.7% | 99.4% | 99.5% |

\*The original detector was evaluated on a random beat-wise split, where beats from the same patient
can appear in both training and test sets. Treat that row as optimistic; the two inter-patient rows
are the fair comparison. Images in the second row are **synthetic renderings** of real beats, not
photographs.

Per-class details and confusion matrices are in `models/*_metrics.json` and `models/*_confusion.png`.

### Computer vision: U-Net vs classical digitizer
Correlation of the recovered trace with the ground-truth trace (60 **synthetic** test images per
condition, real MIT-BIH test beats). Source: `models/cv_eval.md`.

| Condition | Classical (HSV threshold) | U-Net |
|---|---|---|
| clean | 0.754 | 0.999 |
| noisy | 0.787 | 0.999 |
| tilted | 0.804 | 0.998 |
| shadow | 0.784 | 0.999 |
| clutter | 0.433 | 0.998 |
| hard | 0.476 | 0.997 |

### Model comparison (3 seeds, inter-patient test)
Source: `models/benchmark_table.md`.

| Model | Accuracy | PVC (V) sensitivity | PVC (V) precision |
|---|---|---|---|
| Random Forest | 0.912 | 0.853 | 0.949 |
| SVM (RBF) | 0.853 | 0.945 | 0.732 |
| 1D-CNN (signal only) | 0.873 | 0.869 | 0.715 |
| **1D-CNN + R-R (ours)** | 0.924 | 0.948 | 0.866 |
| 1D-ResNet + R-R | 0.932 | 0.950 | 0.880 |
| Conv-BiLSTM + R-R | 0.932 | 0.864 | 0.864 |

---

## Explainable AI

Two independent layers, so each cross-checks the other:

1. **Grad-CAM** on the last Conv1D layer shows *where* in the beat the CNN focused. A deletion test
   (`src/dashboard_data.py`, tab "XAI faithfulness") checks that hiding the highlighted region
   really changes the prediction.
2. **Rule-based clinical reasoning** checks the textbook PVC signs: absent P wave, widened QRS and
   discordant T wave. On image-read beats the QRS width is judged as a **ratio to the strip's own
   confident normal beats**, because absolute milliseconds are not reliable after digitization.

When the CNN is borderline but the rules disagree, the report flags the beat for review instead of
silently trusting one number.

---

## Web app (`app_pro.py`)

```bash
streamlit run app_pro.py
```

| Tab | What it shows |
|---|---|
| Analyze ECG | Upload an image or pick a gallery strip; beats coloured N / S / V / Uncertain, heart rate, patterns, per-beat Grad-CAM + reasons, PDF download |
| Rhythm & HRV | R-R intervals, pNN50, Poincare plot (SD1/SD2), R-R distribution, ectopic timing |
| How it works | The CV stage made visible: U-Net trace, grid calibration, detected R peaks |
| Performance | Labelled numbers for raw signals vs rendered images, and the limits |
| Evaluation lab | Interactive threshold trade-off and per-patient results on real test patients |
| XAI faithfulness | Grad-CAM deletion test |
| Single beat | The original single-beat PVC / Normal detector |

`app.py` is the original 4-tab PVC-only interface (single beat, multi-beat strip, performance, saved reports).

---

## Project structure

```
.
├── app_pro.py                    # multi-class Streamlit app (7 tabs)
├── app.py                        # original PVC-only Streamlit app
├── src/
│   ├── aami_dataset.py           # builds the 4-class AAMI dataset with the inter-patient split
│   ├── train_arrhythmia.py       # trains the 4-class CNN (signal-only or signal + R-R)
│   ├── build_image_dataset.py    # image-domain training set (rendered images of real beats)
│   ├── arrhythmia_pipeline.py    # end-to-end image -> annotated strip pipeline
│   ├── beat_detect.py            # R-peak detector
│   ├── rhythm_analytics.py       # heart rate, couplets, runs, bigeminy, trigeminy
│   ├── report_pdf.py             # PDF report
│   ├── dash_tabs.py, dashboard_data.py   # dashboards and their precomputed data
│   ├── robustness_ablation.py    # raw signal vs image robustness study
│   ├── make_demo_gallery.py      # demo strips + clean-image pipeline evaluation
│   ├── download_ds2.py           # downloads the needed MIT-BIH records
│   ├── cv_unet/                  # U-Net trace segmentation, grid calibration, paper crop
│   └── (original PVC pipeline)   # data_loader, ecg_image_gen, image_to_signal, model, train,
│                                 #   xai_explain, predict_pipeline, multi_beat_pipeline
├── models/                       # trained models, metrics JSON, benchmark and CV evaluation
├── requirements.txt
└── data/                         # MIT-BIH records and generated images (not in git)
```

---

## Setup

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

### Data
MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). The helper
downloads only the records used (about 2.4 MB each):

```bash
python src/download_ds2.py --ds1          # DS1 (training) and DS2 (test) patients into data/mitdb
```

### Rebuild datasets and models (optional; trained models are already in `models/`)
```bash
python src/aami_dataset.py --data_dir data/mitdb --out aami_dataset.npz
python src/train_arrhythmia.py --data aami_dataset.npz --use_rr \
    --out models/arrhythmia_rr.keras --metrics models/arrhythmia_rr_metrics.json \
    --confusion_png models/arrhythmia_rr_confusion.png
python src/cv_unet/train_unet.py --bank aami_dataset.npz
python src/build_image_dataset.py --data_dir data/mitdb --split_from aami_dataset.npz
```
The large `.npz` datasets are not stored in git; the commands above regenerate them.

### Demo gallery and dashboard data
```bash
python src/make_demo_gallery.py --data_dir data/mitdb --models_dir models --out_dir data/ecg_images/gallery
python src/dashboard_data.py              # needs models/image_domain.npz
```

### Original PVC-only pipeline
```bash
python src/train.py --mode real --records 100 101 106 119 200 201 203 208 210 213 215 219 221 223 233 --epochs 25
python src/predict_pipeline.py --image path/to/beat.png --model models/pvc_cnn.keras
python src/multi_beat_pipeline.py --image path/to/strip.png --model models/pvc_cnn.keras
```

---

## Key design decisions

- **Image input, not raw signal:** real-world ECGs often exist only as printouts, scans or screenshots.
- **Lead II:** the standard lead for rhythm monitoring, and MIT-BIH is mainly MLII.
- **1D CNN, not LSTM/Transformer:** beats are short, fixed-length and shape-driven; CNNs train fast
  and Grad-CAM is well established on them. Benchmarked against ResNet and BiLSTM variants above.
- **R-R interval features:** S and V beats are defined by how *early* they arrive, which a 200-sample
  window alone cannot show. Adding R-R features raised accuracy from 0.873 to 0.924 in the benchmark.
- **U-Net digitizer instead of colour thresholding:** the classical HSV method degrades on noise,
  shadows and clutter; the U-Net stays above 0.99 correlation in the same tests.
- **Inter-patient evaluation:** beats from one patient are never split across train and test.

---

## Known limitations

- **Weaker classes:** S (PAC-type) beats are only partly detected (recall about 35-37%) and **F
  (fusion) beats are not reliably detected** (recall near 0%); both are rare and concentrated in a
  few patients. PVC (V) is the strongest class. In calibrated mode the F output is ignored.
- **Image-based accuracy is lower than raw-signal accuracy** (see the results table), and
  image-based PVC counts can run high. No PVC burden percentage is shown for a short strip.
- **Images are synthetic renderings** of real MIT-BIH beats; robustness on real photographs has
  only been checked by eye (`src/cv_unet/try_on_image.py`).
- **Grid calibration needs visible grid lines**; with only the 5 mm lines visible the heart rate can
  come out wrong by a factor of five.
- **Single-lead rhythm strips only** (trained on Lead II), not 12-lead diagnostic sheets.
- **Detection covers only the beats present in the image**; it cannot see beats outside the strip.
- The R-peak detector was tuned on five MIT-BIH records; validate it on more before quoting a number.
- Not a medical device; outputs describe patterns of detected beats and are not diagnoses.

---

## Tech stack

Python · TensorFlow/Keras · OpenCV · SciPy · NumPy · scikit-learn · wfdb · Matplotlib · Pillow ·
ReportLab · Streamlit
