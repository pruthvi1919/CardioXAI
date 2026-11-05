Models and metrics

Trained models and the result files they produced. Large datasets (`aami_dataset.npz`,
`image_domain.npz`) are not stored in git; see the main README for how to regenerate them.

Trained models

| File | What it is |
|---|---|
| `pvc_cnn.keras` | Original binary Normal vs PVC 1D CNN (used by `app.py` and the shape-only mode) |
| `arrhythmia_cnn.keras` | 4-class AAMI CNN (N, S, V, F), raw signal only |
| `arrhythmia_rr.keras` | 4-class CNN + R-R interval features, trained on raw signals |
| `arrhythmia_rr_img.keras` | Same architecture trained on image-domain beats; the classifier `app_pro.py` runs |
| `unet_trace.keras` | U-Net that segments the ECG trace from an image |

Results

| File | Contents |
|---|---|
| `train_metrics.json` | Original binary PVC detector metrics |
| `arrhythmia_metrics.json`, `arrhythmia_rr_metrics.json`, `arrhythmia_rr_img_metrics.json` | Per-class metrics and confusion matrices for the 4-class models (inter-patient test set) |
| `*_confusion.png` | Confusion matrix plots |
| `benchmark_table.md`, `benchmark_results.json`, `benchmark.png` | Comparison against Random Forest, SVM, ResNet and BiLSTM |
| `cv_eval.md`, `cv_eval.json`, `cv_eval_examples.png` | U-Net vs classical digitizer on synthetic test images |
| `robustness_ablation.md` / `.json` (and `ablation_*` folders) | Raw signal vs image-input robustness study |
| `xai_faithfulness.json`, `eval_test_probs.npz` | Data behind the XAI faithfulness and Evaluation lab dashboards |
| `unet_history.json`, `unet_preview.png` | U-Net training history and sample predictions |

All beats and labels are real MIT-BIH data. Image-based results use synthetic renderings of real beats, not photographs.
