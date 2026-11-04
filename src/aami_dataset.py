"""
aami_dataset.py
================
Builds the 4-class AAMI heartbeat dataset (N, S, V, F) from MIT-BIH with the
standard INTER-PATIENT split (de Chazal et al.): DS1 for training, DS2 for
testing, so no patient ever appears in both.

AAMI classes used here:
    N : normal and bundle-branch-block / escape beats   (N, L, R, e, j)
    S : supraventricular ectopic beats                  (A, a, J, S)
    V : ventricular ectopic beats (PVC)                 (V, E)
    F : fusion of ventricular and normal beat           (F)
The AAMI 'Q' class (paced / unclassifiable) is excluded, and so are the four
paced records (102, 104, 107, 217), which is the usual convention.

Each beat is a 200-sample window (90 before + 110 after the R peak at 360 Hz),
the same convention as data_loader.py, so the existing CNN input size and
Grad-CAM code keep working.

Also stored per beat: R-R interval features (previous, next, local mean) for
the optional multi-beat feature experiment, plus the record id so every split
can be checked patient-wise.

Usage:
    from aami_dataset import build_dataset, make_splits, summarize, save_dataset
    data = build_dataset("data/mitdb")
    splits = make_splits(data)
    summarize(data, splits)
    save_dataset("aami_dataset.npz", data, splits)
"""

import os
import numpy as np

SAMPLING_RATE = 360
WINDOW_BEFORE = 90
WINDOW_AFTER = 110
BEAT_LEN = WINDOW_BEFORE + WINDOW_AFTER

CLASS_NAMES = ["N", "S", "V", "F"]

AAMI_MAP = {}
for _s in ["N", "L", "R", "e", "j"]:
    AAMI_MAP[_s] = 0
for _s in ["A", "a", "J", "S"]:
    AAMI_MAP[_s] = 1
for _s in ["V", "E"]:
    AAMI_MAP[_s] = 2
AAMI_MAP["F"] = 3

# Every annotation symbol that marks a heartbeat (used for R-R intervals, so
# paced / unclassifiable beats still count as beats even though they are not
# kept as training samples).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Standard inter-patient split (de Chazal et al., 2004). 22 records each.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]
ALL_RECORDS = DS1 + DS2


def load_record(record_name, data_dir="data/mitdb"):
    """
    Load one record and return its labelled beats, or None if the record has
    no MLII channel. The channel is chosen BY NAME (not assumed to be channel
    0), because in some records (e.g. 114) the lead order is swapped.
    """
    import wfdb

    path = os.path.join(data_dir, record_name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    if "MLII" not in rec.sig_name:
        return None
    signal = rec.p_signal[:, rec.sig_name.index("MLII")]
    n = len(signal)

    # R-R intervals from ALL beat annotations, in order
    beat_samples = np.array(
        [s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS],
        dtype=np.int64)
    rr = np.diff(beat_samples).astype(np.float32)
    prev_rr = {}
    next_rr = {}
    local_rr = {}
    for i, s in enumerate(beat_samples):
        p = rr[i - 1] if i > 0 else (rr[i] if len(rr) else np.nan)
        q = rr[i] if i < len(rr) else (rr[i - 1] if len(rr) else np.nan)
        lo = max(0, i - 10)
        window = rr[lo:i] if i > 0 else rr[:1]
        prev_rr[s] = p
        next_rr[s] = q
        local_rr[s] = float(np.mean(window)) if len(window) else np.nan

    X, y, pos, f_prev, f_next, f_local = [], [], [], [], [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI_MAP:
            continue
        start, end = s - WINDOW_BEFORE, s + WINDOW_AFTER
        if start < 0 or end > n:
            continue
        X.append(signal[start:end])
        y.append(AAMI_MAP[sym])
        pos.append(s)
        f_prev.append(prev_rr.get(s, np.nan))
        f_next.append(next_rr.get(s, np.nan))
        f_local.append(local_rr.get(s, np.nan))

    if not X:
        return None
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "pos": np.asarray(pos, dtype=np.int64),
        "rr_prev": np.asarray(f_prev, dtype=np.float32),
        "rr_next": np.asarray(f_next, dtype=np.float32),
        "rr_local": np.asarray(f_local, dtype=np.float32),
    }


def build_dataset(data_dir="data/mitdb", records=None, verbose=True):
    """Load every record and concatenate into one dataset dict."""
    records = list(records) if records is not None else ALL_RECORDS
    parts = {k: [] for k in ["X", "y", "pos", "rr_prev", "rr_next", "rr_local"]}
    rec_ids = []
    skipped = []
    for r in records:
        try:
            d = load_record(r, data_dir)
        except Exception as e:  # missing file, corrupt file, etc.
            skipped.append((r, repr(e)))
            continue
        if d is None:
            skipped.append((r, "no MLII channel or no usable beats"))
            continue
        for k in parts:
            parts[k].append(d[k])
        rec_ids.extend([r] * len(d["y"]))
        if verbose:
            counts = np.bincount(d["y"], minlength=4)
            print(f"  {r}: {len(d['y']):5d} beats  "
                  + "  ".join(f"{c}={n}" for c, n in zip(CLASS_NAMES, counts)))
    if not parts["X"]:
        raise FileNotFoundError(
            f"No usable records found in '{data_dir}'. Download them first.")
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    data["record"] = np.asarray(rec_ids)
    # R-R features can be NaN for a record with a single beat; make them finite
    for k in ["rr_prev", "rr_next", "rr_local"]:
        col = data[k]
        finite = np.isfinite(col)
        col[~finite] = np.median(col[finite]) if finite.any() else 0.0
    if skipped and verbose:
        print("Skipped:", skipped)
    data["skipped"] = skipped
    return data


def choose_val_records(data, ds1=None, n_val=4, trials=4000, seed=0,
                       target_share=0.18):
    """
    Pick n_val DS1 records for validation (patient-wise, never split inside a
    record). Random search for the subset whose class shares are closest to
    `target_share` of each class's DS1 total, requiring every class to be
    present, so the validation set can actually score S and F beats.
    """
    ds1 = list(ds1) if ds1 is not None else DS1
    present = [r for r in ds1 if np.any(data["record"] == r)]
    counts = np.stack([np.bincount(data["y"][data["record"] == r], minlength=4)
                       for r in present])
    totals = counts.sum(axis=0).astype(float)
    totals[totals == 0] = 1.0
    rng = np.random.default_rng(seed)
    best, best_loss = None, np.inf
    for _ in range(trials):
        idx = rng.choice(len(present), size=min(n_val, len(present)),
                         replace=False)
        sel = counts[idx].sum(axis=0)
        if np.any(sel == 0) and np.all(counts.sum(axis=0) > 0):
            continue
        loss = np.abs(sel / totals - target_share).sum()
        if loss < best_loss:
            best, best_loss = idx, loss
    if best is None:
        best = np.arange(min(n_val, len(present)))
    return sorted(present[i] for i in best)


def make_splits(data, n_val=4, seed=0):
    """
    Returns boolean masks: train (DS1 minus validation records), val (a few
    DS1 records, patient-wise), test (DS2). No record appears in two splits.
    """
    rec = data["record"]
    in_ds1 = np.isin(rec, DS1)
    in_ds2 = np.isin(rec, DS2)
    val_records = choose_val_records(data, n_val=n_val, seed=seed)
    val = in_ds1 & np.isin(rec, val_records)
    train = in_ds1 & ~val
    test = in_ds2
    assert not np.any(train & val) and not np.any(train & test) \
        and not np.any(val & test)
    return {"train": train, "val": val, "test": test,
            "val_records": val_records}


def summarize(data, splits):
    """Print per-split, per-class counts so the split can be sanity-checked."""
    print(f"\n{'split':<7}{'beats':>8}" + "".join(f"{c:>8}" for c in CLASS_NAMES))
    for name in ["train", "val", "test"]:
        m = splits[name]
        counts = np.bincount(data["y"][m], minlength=4)
        print(f"{name:<7}{int(m.sum()):>8}" + "".join(f"{n:>8}" for n in counts))
    print("validation records:", splits["val_records"])
    for name in ["train", "val", "test"]:
        recs = sorted(set(data["record"][splits[name]]))
        print(f"{name} records ({len(recs)}):", " ".join(recs))


def save_dataset(path, data, splits):
    """Save everything needed for training/evaluation to one .npz file."""
    split_id = np.full(len(data["y"]), -1, dtype=np.int8)
    split_id[splits["train"]] = 0
    split_id[splits["val"]] = 1
    split_id[splits["test"]] = 2
    np.savez_compressed(
        path, X=data["X"], y=data["y"], record=data["record"],
        pos=data["pos"], rr_prev=data["rr_prev"], rr_next=data["rr_next"],
        rr_local=data["rr_local"], split=split_id,
        class_names=np.array(CLASS_NAMES))
    print(f"Saved {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_saved(path):
    """Load a dataset saved by save_dataset and rebuild the split masks."""
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in ["X", "y", "record", "pos", "rr_prev",
                              "rr_next", "rr_local"]}
    splits = {"train": z["split"] == 0, "val": z["split"] == 1,
              "test": z["split"] == 2}
    return data, splits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mitdb")
    ap.add_argument("--out", default="aami_dataset.npz")
    args = ap.parse_args()
    d = build_dataset(args.data_dir)
    s = make_splits(d)
    summarize(d, s)
    save_dataset(args.out, d, s)
