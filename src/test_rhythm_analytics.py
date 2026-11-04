"""Run: python test_rhythm_analytics.py   (all asserts use sequences with known answers)"""
import numpy as np
from rhythm_analytics import analyze, summary_text

def seq(s, rr=0.8, fs=360):
    lab = list(s); r = np.cumsum([0] + [int(rr * fs)] * (len(lab) - 1)); return lab, r

# 1 plain normal rhythm at 75 bpm
l, r = seq("N" * 20, rr=0.8); a = analyze(l, r)
assert a["heart_rate"]["mean_bpm"] == 75.0 and a["pvc_burden_pct"] == 0 and a["V"]["total"] == 0

# 2 isolated + couplet + run of 4 (fast) + bigeminy (4 PVCs) + trigeminy (3 PVCs)
s = "NNNVNNN" + "VV" + "NNN" + "VVVV" + "NNN" + "VNVNVNVNV"[:7] + "NN" + "NNVNNVNNV" + "NN"
l, r = seq(s); a = analyze(l, r); V = a["V"]
print(s); print({k: V[k] for k in V if k != "runs_ge3"}, V["runs_ge3"])
assert V["isolated"] >= 1 and V["couplets"] == 1
assert len(V["runs_ge3"]) == 1 and V["runs_ge3"][0]["beats"] == 4
assert V["bigeminy_episodes"] == 1 and V["bigeminy_ectopics"] == 4
assert V["trigeminy_episodes"] == 1 and V["trigeminy_ectopics"] == 3

# 3 fast run flagged >100 bpm, slow run not
l = list("NNN" + "VVV" + "NNN"); r = [0, 288, 576]
r += [r[-1] + 288 * (i + 1) for i in range(3)]; r = list(r)
r += [r[-1] + 288 * (i + 1) for i in range(3)]
l2, r2 = list("NNNVVVNNN"), np.array([0, 288, 576, 720, 864, 1008, 1296, 1584, 1872])  # 0.4 s inside run = 150 bpm
a = analyze(l2, r2); assert a["V"]["runs_ge3"][0]["faster_than_100"] is True
l3, r3 = seq("NNNVVVNNN", rr=0.8); assert analyze(l3, r3)["V"]["runs_ge3"][0]["faster_than_100"] is False

# 4 pause detection + gap exclusion
r4 = np.array([0, 288, 576, 576 + 360 * 2.5, 576 + 360 * 2.5 + 288]).astype(int)
a = analyze(list("NNNNN"), r4); assert a["pauses_ge_2s"] == 1

# 5 S patterns and label forms (ints and strings)
l, r = seq("NNANNN".replace("A", "S") + "SS" + "NN"); a = analyze(l, r)
assert a["S"]["isolated"] == 1 and a["S"]["couplets"] == 1
assert analyze([0, 2, 0, 2, 0, 2, 0], seq("N" * 7)[1])["V"]["bigeminy_episodes"] == 1

# 6 burden flags and edge cases
l, r = seq("N" * 90 + "V" * 10); assert analyze(l, r)["pvc_burden_flag"].startswith("moderate")
assert "Not enough" in summary_text(analyze(["N"], [0]))
assert analyze(["N", "V"], [100, 0])["counts"]["V"] == 1          # unsorted input is sorted
print(summary_text(analyze(*seq(s))))
print("ALL TESTS PASSED")
