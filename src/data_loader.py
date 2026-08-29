"""
data_loader.py
================
Loads MIT-BIH Arrhythmia Database records and extracts labeled heartbeat
segments (Normal vs PVC) for training.

TWO MODES:
1. REAL MODE  - reads local MIT-BIH files (.dat/.hea/.atr) that YOU download
                from PhysioNet: https://physionet.org/content/mitdb/1.0.0/
                Put them in: data/mitdb/  (e.g. data/mitdb/100.dat, 100.hea, 100.atr)
2. DEMO MODE  - generates synthetic ECG-like beats so the whole pipeline can
                be smoke-tested without internet access. Use this only to
                verify your code runs end-to-end, NOT for your real results.

MIT-BIH annotation symbols we care about:
    'N' -> Normal beat
    'V' -> Premature Ventricular Contraction (PVC)   <-- our positive class
(Other symbols are excluded to keep this a clean binary problem. You can
extend to multi-class later if your project asks for it.)
"""

import os
import numpy as np

SAMPLING_RATE = 360          # MIT-BIH standard
WINDOW_BEFORE = 90           # samples before R-peak
WINDOW_AFTER = 110           # samples after R-peak
BEAT_LEN = WINDOW_BEFORE + WINDOW_AFTER

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
PVC_SYMBOLS = {'V', 'E'}


def list_local_records(data_dir="data/mitdb"):
    """Return list of record names available locally (based on .hea files)."""
    if not os.path.isdir(data_dir):
        return []
    return sorted({f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea")})


def load_record_real(record_name, data_dir="data/mitdb"):
    """
    Load one MIT-BIH record using wfdb and extract beat-centered windows.
    Requires record_name.dat / .hea / .atr already present in data_dir.
    """
    import wfdb

    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]  # use lead 0 (MLII in most records)

    beats, labels = [], []
    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in NORMAL_SYMBOLS and symbol not in PVC_SYMBOLS:
            continue
        start = sample_idx - WINDOW_BEFORE
        end = sample_idx + WINDOW_AFTER
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        beats.append(beat)
        labels.append(1 if symbol in PVC_SYMBOLS else 0)

    return np.array(beats), np.array(labels)


def load_dataset_real(record_names=None, data_dir="data/mitdb"):
    """Load and concatenate multiple MIT-BIH records."""
    if record_names is None:
        record_names = list_local_records(data_dir)
    if not record_names:
        raise FileNotFoundError(
            f"No MIT-BIH records found in '{data_dir}'. Download from "
            f"https://physionet.org/content/mitdb/1.0.0/ and place the "
            f".dat/.hea/.atr files there, or use load_dataset_demo() to test."
        )

    all_beats, all_labels = [], []
    for rec in record_names:
        try:
            beats, labels = load_record_real(rec, data_dir)
            all_beats.append(beats)
            all_labels.append(labels)
            print(f"  loaded {rec}: {len(labels)} beats "
                  f"({int(labels.sum())} PVC, {int((labels == 0).sum())} normal)")
        except Exception as e:
            print(f"  skipping {rec}: {e}")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def _synthetic_beat(is_pvc, rng):
    """
    Generate one fake-but-plausible beat for offline pipeline testing.
    Normal beat: clean P-QRS-T with narrow QRS.
    PVC beat: no P wave, wide/bizarre QRS, T wave deflected opposite to QRS.
    """
    t = np.linspace(-1, 1, BEAT_LEN)
    beat = np.zeros(BEAT_LEN)

    def gauss(center, width, amp):
        return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    if not is_pvc:
        beat += gauss(-0.55, 0.03, 0.15)     # P wave
        beat += gauss(0.0, 0.012, 1.0)       # QRS (narrow, tall)
        beat += gauss(0.35, 0.08, 0.3)       # T wave (same direction)
    else:
        # no P wave
        beat += gauss(0.0, 0.05, -0.9)       # wide, bizarre QRS
        beat += gauss(0.15, 0.05, 0.5)
        beat += gauss(0.4, 0.09, -0.35)      # T wave inverted (opposite polarity)

    beat += rng.normal(0, 0.02, size=BEAT_LEN)  # noise
    return beat


def load_dataset_demo(n_per_class=400, seed=42):
    """Synthetic dataset for smoke-testing the pipeline without internet."""
    rng = np.random.default_rng(seed)
    beats, labels = [], []
    for _ in range(n_per_class):
        beats.append(_synthetic_beat(False, rng))
        labels.append(0)
        beats.append(_synthetic_beat(True, rng))
        labels.append(1)
    X = np.array(beats)
    y = np.array(labels)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    print("Local MIT-BIH records found:", list_local_records())
    X, y = load_dataset_demo(n_per_class=5)
    print("Demo dataset shape:", X.shape, y.shape)
