#!/usr/bin/env python3
"""
recovar_playback — Detect picks in miniSEED files using scautopick and
score each pick with RecovAR. Outputs a CSV of scored picks.

Usage:
    seiscomp exec recovar_playback --input /path/to/mseeds
    seiscomp exec recovar_playback --input /path/to/mseeds --output results.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from obspy import UTCDateTime, read, Stream
from scipy.signal import resample as sp_resample

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from recovar_scorer import RecovARScorer

MODEL_PATH    = os.path.join(_REPO, "models", "representation_cross_covariances.h5")
DB_URL        = "mysql://sysop:sysop@localhost/seiscomp"
WINDOW_BEFORE = 5.0
WINDOW_AFTER  = 25.0
TARGET_FS     = 100
N_SAMPLES     = 3000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_picks(xml_text):
    """Parse picks from scautopick --ep SCML output."""
    picks = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return picks

    for elem in root.iter():
        if elem.tag.split("}")[-1] != "pick":
            continue
        try:
            time_val = next(
                UTCDateTime(sub.text)
                for child in elem
                for sub in child
                if child.tag.split("}")[-1] == "time"
                and sub.tag.split("}")[-1] == "value"
            )
            wid = next(
                child for child in elem
                if child.tag.split("}")[-1] == "waveformID"
            )
            picks.append({
                "id":  elem.get("publicID", ""),
                "time": time_val,
                "net": wid.get("networkCode", ""),
                "sta": wid.get("stationCode", ""),
                "loc": wid.get("locationCode", ""),
                "cha": wid.get("channelCode", ""),
            })
        except StopIteration:
            continue
    return picks


def extract_waveform(stream, pick):
    """Return a (N_SAMPLES, 3) float32 array centred on the pick, or None."""
    t0    = pick["time"]
    t_s   = t0 - WINDOW_BEFORE
    t_e   = t0 + WINDOW_AFTER
    band  = pick["cha"][:2]
    net, sta, loc = pick["net"], pick["sta"], pick["loc"]

    def best(chars):
        for c in chars:
            candidates = [
                tr for tr in stream
                if tr.stats.network  == net
                and tr.stats.station  == sta
                and tr.stats.location == loc
                and tr.stats.channel.startswith(band)
                and tr.stats.channel.endswith(c)
            ]
            if candidates:
                return candidates[0]
        return None

    z  = best(["Z"])
    h1 = best(["N", "1"])
    h2 = best(["E", "2"])
    if any(c is None for c in (z, h1, h2)):
        return None

    def to_array(tr):
        sliced = tr.slice(t_s, t_e)
        if sliced is None or sliced.stats.npts == 0:
            return np.zeros(N_SAMPLES, dtype=np.float64)
        arr = sliced.data.astype(np.float64)
        fs  = sliced.stats.sampling_rate
        if abs(fs - TARGET_FS) > 0.01:
            n_out = int(round(len(arr) * TARGET_FS / fs))
            arr   = sp_resample(arr, n_out)
        return arr[:N_SAMPLES] if len(arr) >= N_SAMPLES else np.pad(arr, (0, N_SAMPLES - len(arr)))

    return np.stack([to_array(z), to_array(h1), to_array(h2)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Score seismic picks detected by scautopick using RecovAR.")
    p.add_argument("--input",  required=True,
                   help="Folder containing miniSEED files (.mseed / .ms / .miniseed)")
    p.add_argument("--output", default="scored_picks.csv",
                   help="Output CSV file (default: scored_picks.csv)")
    return p.parse_args()


def main():
    args   = parse_args()
    indir  = os.path.expanduser(args.input)

    if not os.path.isdir(indir):
        sys.exit(f"Input folder not found: {indir}")

    files = sorted(
        os.path.join(indir, f) for f in os.listdir(indir)
        if f.lower().endswith((".mseed", ".ms", ".miniseed"))
    )
    if not files:
        sys.exit(f"No miniSEED files found in {indir}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"[1/4] Loading RecovAR model...")
    scorer = RecovARScorer(MODEL_PATH)
    print( "      Model ready")

    # ── 2. Concatenate files for scautopick ───────────────────────────────────
    print(f"[2/4] Found {len(files)} miniSEED file(s) — running scautopick...")
    tmp = tempfile.NamedTemporaryFile(suffix=".mseed", delete=False)
    tmp.close()
    with open(tmp.name, "wb") as out:
        for path in files:
            with open(path, "rb") as f:
                out.write(f.read())

    result = subprocess.run(
        f"seiscomp exec scautopick --ep --playback -I file://{tmp.name} -d {DB_URL}",
        shell=True, capture_output=True, text=True,
    )
    os.unlink(tmp.name)

    picks = parse_picks(result.stdout)
    print(f"      {len(picks)} pick(s) detected")
    if not picks:
        print("      Nothing to score.")
        return

    # ── 3. Load waveforms ─────────────────────────────────────────────────────
    print(f"[3/4] Loading waveforms...")
    stream = Stream()
    for path in files:
        stream += read(path)
    stream.merge(method=1, fill_value=0)

    # ── 4. Score each pick ────────────────────────────────────────────────────
    print(f"[4/4] Scoring {len(picks)} pick(s)...")
    rows = []
    for pick in picks:
        waveform = extract_waveform(stream, pick)
        if waveform is None:
            print(f"      SKIP {pick['id']} — waveform unavailable")
            continue
        score = scorer.score(waveform)
        rows.append((pick["id"], str(pick["time"]),
                     pick["net"], pick["sta"], pick["loc"], pick["cha"],
                     f"{score:.4f}"))
        print(f"      {pick['net']}.{pick['sta']}.{pick['loc']}.{pick['cha']}"
              f"  {pick['time']}  score={score:.4f}")

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pick_id", "pick_time", "net", "sta", "loc", "cha", "score"])
        w.writerows(rows)

    print(f"\nDone. {len(rows)} scored pick(s) → {args.output}")


if __name__ == "__main__":
    main()
