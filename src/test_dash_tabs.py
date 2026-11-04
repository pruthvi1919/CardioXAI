"""Tests for the pure-math helpers in dash_tabs.py. Run: py -3.12 src/test_dash_tabs.py  (from project root)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dash_tabs import hrv_metrics, poincare_sd, ectopic_timing, confusion_at, per_record_table

# HRV: perfectly regular 0.8 s rhythm -> SDNN 0, RMSSD 0, HR 75
h = hrv_metrics([0.8] * 8)
assert abs(h["hr_bpm"] - 75) < 1e-9 and h["sdnn_ms"] == 0 and h["rmssd_ms"] == 0 and h["pnn50_pct"] == 0
# hand-computed: rr = .8,.9,.8,.9 -> diffs .1,-.1,.1 -> RMSSD 100 ms; sample SD of rr = 57.735 ms; pNN50 = 100%
h = hrv_metrics([0.8, 0.9, 0.8, 0.9])
assert abs(h["rmssd_ms"] - 100) < 1e-6 and abs(h["sdnn_ms"] - 57.7350269) < 1e-4 and h["pnn50_pct"] == 100
assert hrv_metrics([0.8, 0.9]) is None
# Poincare: alternating 800/900 -> points (800,900),(900,800),(800,900): SD1 large, SD2 small
sd1, sd2 = poincare_sd([0.8, 0.9, 0.8, 0.9, 0.8])
assert sd1 > sd2
# Ectopic timing: N N N V N N, normal RR 0.8; V comes early (0.5) then pause 1.1 -> sum 1.6 (=2.0*0.8)
t = np.cumsum([0, 0.8, 0.8, 0.5, 1.1, 0.8])
rows, base = ectopic_timing(["N", "N", "N", "V", "N", "N"], t)
assert abs(base - 0.8) < 1e-9 and len(rows) == 1
r = rows[0]
assert r["beat"] == 4 and r["premature"] and r["full_comp_pause"]
assert abs(r["coupling_ratio"] - 0.625) < 1e-9 and abs(r["pause_ratio"] - 1.375) < 1e-9
# a V with no N on one side is skipped; no normal reference -> (rows, None)
assert ectopic_timing(["V", "N", "N"], [0, 1, 2])[0] == []
assert ectopic_timing(["V", "V", "V"], [0, 1, 2])[1] is None

# confusion_at: tiny hand-made example. 4 beats true N,N,V,F(ignored); probs over (N,S,V,F)
prob = np.array([[.9, .05, .04, .01], [.4, .35, .2, .05], [.1, .1, .75, .05], [.2, .2, .2, .4]], float)
y = np.array([0, 0, 2, 3])
a = confusion_at(prob, y, 0.5)
assert a["cm"].tolist() == [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]]       # beat 2 uncertain, F dropped
assert abs(a["coverage"] - 2 / 3) < 1e-9 and a["accuracy"] == 1.0
assert a["V_precision"] == 1.0 and a["V_sensitivity"] == 1.0 and abs(a["N_sensitivity"] - 0.5) < 1e-9
b = confusion_at(prob, y, 0.0)
assert b["cm"].tolist() == [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]] and b["accuracy"] == 1.0
rows = per_record_table(prob, y, np.array(["a", "a", "b", "b"]), 0.5)
assert [r["record"] for r in rows] == ["a", "b"] and rows[0]["beats"] == 2 and rows[1]["beats"] == 1
print("helpers OK")

# real data cross-check against the numbers already reported for the image-domain model (threshold 0.6)
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "eval_test_probs.npz")
if os.path.exists(p):
    d = np.load(p)
    c = confusion_at(d["prob"].astype(np.float32), d["y"].astype(int), 0.6)
    print({k: round(v, 3) for k, v in c.items() if k != "cm"})
    assert abs(c["coverage"] - 0.893) < 0.002 and abs(c["accuracy"] - 0.875) < 0.002
    assert abs(c["V_precision"] - 0.691) < 0.002 and abs(c["V_sensitivity"] - 0.761) < 0.002
    print("matches the earlier real-data numbers (REAL beats, image-domain) OK")
