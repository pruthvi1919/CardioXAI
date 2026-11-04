"""
rhythm_analytics.py
====================
Turns a sequence of classified beats (N / S / V / F + R-peak positions) into
clinically readable rhythm statistics. Pure NumPy, no model needed, so it runs
on the output of ANY stage: the CNN's predictions, or MIT-BIH ground-truth
labels for checking.

    from rhythm_analytics import analyze, summary_text
    res = analyze(labels, r_samples, fs=360)      # labels: ints 0..3 or "N","S","V","F"
    print(summary_text(res))

Reported:
  * heart rate (mean, and min/max over a 5-beat window), RR intervals, pauses >= 2 s
  * beat counts per class
  * ventricular (V) patterns: isolated, couplets, runs of >= 3 (and whether the run is
    faster than 100 bpm), bigeminy, trigeminy
  * the same patterns for S beats (atrial/supraventricular)

PVC / PAC burden % is still computed in analyze() (for evaluation scripts), but it is NOT
shown in the summary: clinically, burden is measured over a 24-hour recording, and the
share of PVCs in one short strip says nothing about it.

These are PATTERN descriptions of the beats we detected, NOT diagnoses.

CLI check against MIT-BIH ground truth, from the saved dataset (no raw files needed):
    python src/rhythm_analytics.py --npz aami_dataset.npz --record 119
or from the raw record (needs wfdb, aami_dataset.py and data/mitdb):
    python src/rhythm_analytics.py --record 119 --data_dir data/mitdb

Note: the .npz only keeps beats with a full 200-sample window, so the first and
last beat of a record can be missing; counts differ from the raw annotations by a beat or two.
"""

import argparse
import numpy as np

CLASS_NAMES = ["N", "S", "V", "F"]
_SYM = {c: i for i, c in enumerate(CLASS_NAMES)}
MIN_RR, MAX_RR = 0.2, 3.0          # s; outside this an interval is a gap, not a beat-to-beat rate
PAUSE_S = 2.0                      # s; clinical "pause" threshold
RUN_RATE_BPM = 100.0               # a run of >= 3 faster than this is flagged "tachycardic"


def _to_int(labels):
    out = []
    for l in labels:
        out.append(_SYM[l] if isinstance(l, str) else int(l))
    return np.asarray(out, dtype=np.int64)


def _runs(mask):
    """[(start, length)] of consecutive True values."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _alternating(labels, cls, every):
    """
    Episodes where a `cls` beat occurs every `every`-th beat with only N beats
    in between (every=2 -> bigeminy N-V-N-V-N-V, every=3 -> trigeminy), at least
    3 ectopics. Returns [(start_index, n_ectopic)].
    """
    n, episodes, i = len(labels), [], 0
    while i < n:
        if labels[i] != cls:
            i += 1
            continue
        k, j = 1, i
        while (j + every < n and labels[j + every] == cls
               and np.all(labels[j + 1:j + every] == 0)):
            k += 1
            j += every
        if k >= 3:
            episodes.append((i, k))
            i = j + 1
        else:
            i += 1
    return episodes


def _pattern_block(labels, rr_after, cls):
    """Isolated / couplet / run / bigeminy / trigeminy stats for one ectopic class."""
    runs = _runs(labels == cls)
    isolated = sum(1 for _, L in runs if L == 1)
    couplets = sum(1 for _, L in runs if L == 2)
    long_runs = []
    for s, L in runs:
        if L >= 3:
            rr = rr_after[s:s + L - 1]            # intervals BETWEEN the run's beats
            rate = 60.0 / np.mean(rr) if len(rr) and np.all(np.isfinite(rr)) else float("nan")
            long_runs.append({"start_beat": int(s), "beats": int(L),
                              "rate_bpm": None if not np.isfinite(rate) else round(float(rate), 1),
                              "faster_than_100": bool(np.isfinite(rate) and rate > RUN_RATE_BPM)})
    bi = _alternating(labels, cls, 2)
    tri = _alternating(labels, cls, 3)
    return {"total": int((labels == cls).sum()), "isolated": isolated, "couplets": couplets,
            "runs_ge3": long_runs, "bigeminy_episodes": len(bi),
            "bigeminy_ectopics": int(sum(k for _, k in bi)),
            "trigeminy_episodes": len(tri), "trigeminy_ectopics": int(sum(k for _, k in tri))}


def _flag(pct):
    return "low (<1%)" if pct < 1 else ("moderate (1-10%)" if pct <= 10 else "high (>10%)")


def analyze(labels, r_samples, fs=360):
    labels = _to_int(labels)
    r = np.asarray(r_samples, dtype=np.float64)
    if len(labels) != len(r):
        raise ValueError("labels and r_samples must have the same length")
    order = np.argsort(r)
    labels, r = labels[order], r[order]
    n = len(labels)
    res = {"fs": fs, "n_beats": int(n)}
    if n < 2:
        res.update(duration_s=0.0, counts={c: int((labels == i).sum()) for i, c in enumerate(CLASS_NAMES)})
        res["note"] = "need at least 2 beats for rhythm statistics"
        return res

    rr = np.diff(r) / fs                              # rr[i] = interval after beat i
    valid = (rr >= MIN_RR) & (rr <= MAX_RR)
    rr_after = np.append(rr, np.nan)                  # one per beat (last has none)
    res["duration_s"] = round(float(r[-1] - r[0]) / fs, 2)
    res["counts"] = {c: int((labels == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    res["intervals_excluded"] = int((~valid).sum())

    if valid.any():
        v = rr[valid]
        hr = 60.0 / v.mean()
        k = min(5, len(v))
        roll = np.convolve(v, np.ones(k) / k, mode="valid")      # 5-beat average RR
        res["heart_rate"] = {"mean_bpm": round(float(hr), 1),
                             "min_bpm": round(float(60.0 / roll.max()), 1),
                             "max_bpm": round(float(60.0 / roll.min()), 1),
                             "rr_mean_s": round(float(v.mean()), 3),
                             "rr_sd_ms": round(float(v.std() * 1000), 1)}
    else:
        res["heart_rate"] = None
    res["pauses_ge_2s"] = int((rr >= PAUSE_S).sum())

    # kept for evaluation scripts only; not shown to the user (see module docstring)
    total = max(n, 1)
    res["pvc_burden_pct"] = round(100.0 * (labels == 2).sum() / total, 2)
    res["s_burden_pct"] = round(100.0 * (labels == 1).sum() / total, 2)
    res["pvc_burden_flag"] = _flag(res["pvc_burden_pct"])
    res["V"] = _pattern_block(labels, rr_after, 2)
    res["S"] = _pattern_block(labels, rr_after, 1)
    return res


def summary_text(res):
    """Plain-English summary for the report / app (no burden %, see module docstring)."""
    if res.get("n_beats", 0) < 2:
        return "Not enough beats for rhythm analysis."
    c, hr = res["counts"], res.get("heart_rate")
    lines = [f"{res['n_beats']} beats over {res['duration_s']} s "
             f"(N {c['N']}, S {c['S']}, V {c['V']}, F {c['F']})."]
    if hr:
        lines.append(f"Heart rate {hr['mean_bpm']} bpm on average "
                     f"(5-beat range {hr['min_bpm']}-{hr['max_bpm']} bpm).")
    if res["pauses_ge_2s"]:
        lines.append(f"{res['pauses_ge_2s']} pause(s) of 2 s or more.")
    V, S = res["V"], res["S"]
    if V["total"]:
        lines.append(f"{V['total']} PVC(s): {V['isolated']} isolated, {V['couplets']} couplet(s)")
        extra = []
        for run in V["runs_ge3"]:
            tag = " at >100 bpm" if run["faster_than_100"] else ""
            extra.append(f"run of {run['beats']} PVCs{tag}")
        if V["bigeminy_episodes"]:
            extra.append(f"{V['bigeminy_episodes']} bigeminy episode(s)")
        if V["trigeminy_episodes"]:
            extra.append(f"{V['trigeminy_episodes']} trigeminy episode(s)")
        lines[-1] += (", " + ", ".join(extra) if extra else "") + "."
    else:
        lines.append("No PVCs detected.")
    if S["total"]:
        txt = (f"{S['total']} supraventricular (PAC-type) beat(s): {S['isolated']} isolated, "
               f"{S['couplets']} couplet(s)")
        if S["runs_ge3"]:
            txt += f", {len(S['runs_ge3'])} run(s) of 3+"
        if S["bigeminy_episodes"]:
            txt += f", {S['bigeminy_episodes']} bigeminy episode(s)"
        lines.append(txt + ".")
    lines.append("Pattern description of detected beats; not a diagnosis.")
    return " ".join(lines)


def from_record(record, data_dir="data/mitdb"):
    """MIT-BIH GROUND-TRUTH labels for one record (to sanity-check the analytics)."""
    from aami_dataset import load_record, SAMPLING_RATE
    d = load_record(str(record), data_dir)
    if d is None:
        raise ValueError(f"record {record} has no MLII channel / no beats")
    return analyze(d["y"], d["pos"], SAMPLING_RATE)


def from_npz(npz_path, record, fs=360):
    """MIT-BIH GROUND-TRUTH labels for one record, read from aami_dataset.npz."""
    z = np.load(npz_path, allow_pickle=False)
    rec = np.array([str(x) for x in z["record"]])
    m = rec == str(record)
    if not m.any():
        raise ValueError(f"record {record} not in {npz_path}; available: "
                         f"{sorted(set(rec.tolist()))[:12]}...")
    return analyze(z["y"][m], z["pos"][m], fs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--npz", default=None, help="aami_dataset.npz (preferred, no raw files needed)")
    ap.add_argument("--data_dir", default="data/mitdb")
    a = ap.parse_args()
    out = from_npz(a.npz, a.record) if a.npz else from_record(a.record, a.data_dir)
    print(summary_text(out))
