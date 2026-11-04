"""
download_ds2.py
================
Downloads ONLY what the ablation / image-domain training need: the 22 MIT-BIH test-patient (DS2)
records, plus the 22 training-patient (DS1) records with --ds1 (.dat + .hea + .atr per record,
about 2.4 MB each) instead of the whole 73 MB database.

  * tries the Amazon-hosted PhysioNet copy first (usually much faster), then physionet.org
  * resumes half-finished files, skips finished ones, 30 s timeouts, 5 retries per file
  * prints  [file n/66]  and bytes, so you can see it is alive

    py -3.12 src/download_ds2.py                 # DS2 only, saves to data/mitdb
    py -3.12 src/download_ds2.py --ds1           # also the DS1 training records
    py -3.12 src/download_ds2.py --out data/mitdb --extra 119,106   # add more records
Run it again if it stops; it continues where it left off.
"""
import os
import sys
import time
import argparse
import requests

DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
MIRRORS = ["https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/",
           "https://physionet.org/files/mitdb/1.0.0/"]
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']          # the 22 training-patient records (paced 102, 104 excluded)
EXT = ["hea", "atr", "dat"]
EXPECTED_MIN = {"dat": 600_000, "atr": 1_000, "hea": 50}      # sanity: reject error pages


def fetch(name, out_dir, bases, retries=5, chunk=64 * 1024, log=print):
    dest = os.path.join(out_dir, name)
    ext = name.rsplit(".", 1)[1]
    if os.path.exists(dest) and os.path.getsize(dest) >= EXPECTED_MIN[ext]:
        return "have"
    part = dest + ".part"
    for attempt in range(retries):
        base = bases[attempt % len(bases)]
        try:
            start = os.path.getsize(part) if os.path.exists(part) else 0
            hdr = {"Range": f"bytes={start}-"} if start else {}
            with requests.get(base + name, headers=hdr, stream=True, timeout=(10, 30)) as r:
                if r.status_code == 416:                      # already complete
                    os.replace(part, dest); return "done"
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                if r.status_code == 200:
                    start = 0                                 # server ignored Range: restart
                with open(part, "ab" if start else "wb") as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
            if os.path.getsize(part) < EXPECTED_MIN[ext]:
                raise IOError("file too small (error page?)")
            os.replace(part, dest)
            return "done"
        except Exception as e:
            log(f"    retry {attempt + 1}/{retries} for {name} via {base.split('/')[2]}: {e}")
            time.sleep(min(2 * (attempt + 1), 8))
    return "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mitdb")
    ap.add_argument("--extra", default="", help="comma-separated extra record names")
    ap.add_argument("--ds1", action="store_true", help="also download the 22 DS1 training records")
    ap.add_argument("--mirror", default=None, help="try this base URL first (ends with /)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = DS2 + (DS1 if a.ds1 else []) + [r for r in a.extra.split(",") if r]
    recs = list(dict.fromkeys(recs))
    bases = ([a.mirror] if a.mirror else []) + MIRRORS
    files = [f"{r}.{e}" for r in recs for e in EXT]
    t0, failed = time.time(), []
    for i, name in enumerate(files, 1):
        t = time.time()
        res = fetch(name, a.out, bases)
        size = os.path.getsize(os.path.join(a.out, name)) / 1e3 if res != "FAILED" else 0
        print(f"[file {i}/{len(files)}] {name:<9} {res:<6} {size:8.0f} KB  {time.time() - t:5.1f}s "
              f"(total {time.time() - t0:4.0f}s)", flush=True)
        if res == "FAILED":
            failed.append(name)
    print("\nDONE" if not failed else f"\nMISSING {len(failed)} files: {failed}\nRun again to retry.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
