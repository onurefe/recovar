#!/usr/bin/env python3
"""
batch_score_test.py — Validate RecovAR scoring over a diverse set of
waveforms fetched live from IRIS FDSN.

Fetches M6.5+ earthquake P-arrivals and quiet noise windows at IU.ANMO,
scores each with RecovARScorer, and prints a summary statistics table.

Usage:
    python3 batch_score_test.py
    python3 batch_score_test.py --output results.csv --events 30
"""

import argparse
import csv
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from scipy.signal import resample as sp_resample

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from recovar_scorer import RecovARScorer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(_REPO, "models", "representation_cross_covariances.h5")

NET, STA, LOC, BAND = "IU", "ANMO", "00", "HH"
STA_LAT,  STA_LON   = 34.9459, -106.4572   # IU.ANMO coordinates

WINDOW_BEFORE = 5.0    # s before P arrival
WINDOW_AFTER  = 25.0   # s after  P arrival
TARGET_FS     = 100    # Hz
N_SAMPLES     = 3000   # 30 s × 100 Hz

N_NOISE_SAMPLES = 10   # synthetic noise waveforms to generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_3c(client, t_start, t_end):
    """
    Fetch Z/N/E (or Z/1/2) from IRIS and return a (N_SAMPLES, 3) float32
    array ordered [Z, N/1, E/2], or None on failure.
    """
    try:
        st = client.get_waveforms(NET, STA, LOC, BAND + "?", t_start, t_end)
    except Exception:
        return None

    def pick_comp(chars):
        for c in chars:
            tr = next((t for t in st if t.stats.channel.endswith(c)), None)
            if tr is not None:
                return tr
        return None

    z  = pick_comp(["Z"])
    h1 = pick_comp(["N", "1"])
    h2 = pick_comp(["E", "2"])

    if any(c is None for c in (z, h1, h2)):
        return None

    def to_array(tr):
        arr = tr.data.astype(np.float64)
        fs  = tr.stats.sampling_rate
        if abs(fs - TARGET_FS) > 0.01:
            n_out = int(round(len(arr) * TARGET_FS / fs))
            arr = sp_resample(arr, n_out)
        if len(arr) >= N_SAMPLES:
            return arr[:N_SAMPLES]
        return np.pad(arr, (0, N_SAMPLES - len(arr)))

    channels = [to_array(tr) for tr in (z, h1, h2)]
    return np.stack(channels, axis=-1).astype(np.float32)


def p_arrival(taup, ev_lat, ev_lon, ev_depth_km):
    """Return theoretical P travel-time in seconds using iasp91."""
    dist_deg = locations2degrees(STA_LAT, STA_LON, ev_lat, ev_lon)
    arrivals = taup.get_travel_times(ev_depth_km, dist_deg, phase_list=["P", "p"])
    if not arrivals:
        return None
    return arrivals[0].time   # seconds after origin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch RecovAR scoring test against IRIS data.")
    p.add_argument("--events",  type=int, default=50,                  help="Number of M6.5+ events to fetch (default: 50)")
    p.add_argument("--output",  default="batch_score_results.csv",     help="Output CSV (default: batch_score_results.csv)")
    p.add_argument("--min-mag", type=float, default=6.5,               help="Minimum magnitude (default: 6.5)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading RecovAR model...")
    scorer = RecovARScorer(MODEL_PATH)
    print("Model loaded.\n")

    iris  = Client("IRIS")
    usgs  = Client("USGS")
    taup  = TauPyModel("iasp91")
    rows  = []   # (label, pick_id, score)

    # ── Earthquakes ──────────────────────────────────────────────────────────
    print(f"Fetching {args.events} M{args.min_mag}+ events from USGS catalog...")
    try:
        catalog = usgs.get_events(
            starttime    = UTCDateTime("2010-01-01"),
            endtime      = UTCDateTime("2024-01-01"),
            minmagnitude = args.min_mag,
            limit        = args.events,
            orderby      = "magnitude",
        )
    except Exception as e:
        sys.exit(f"Failed to fetch catalog: {e}")

    print(f"Got {len(catalog)} events. Scoring...\n")
    print(f"{'#':<4} {'event':<30} {'mag':<6} {'dist_deg':<10} {'result'}")
    print("-" * 70)

    for i, event in enumerate(catalog, 1):
        origin = event.preferred_origin() or event.origins[0]
        mag    = (event.preferred_magnitude() or event.magnitudes[0]).mag
        label  = f"M{mag:.1f} {origin.time.strftime('%Y-%m-%d')}"

        tt = p_arrival(taup, origin.latitude, origin.longitude, (origin.depth or 10000) / 1000)
        if tt is None:
            print(f"{i:<4} {label:<30} {mag:<6.1f} {'no P arrival':>10}  SKIP")
            continue

        p_time = origin.time + tt
        t_start = p_time - WINDOW_BEFORE
        t_end   = p_time + WINDOW_AFTER
        dist    = locations2degrees(STA_LAT, STA_LON, origin.latitude, origin.longitude)

        waveform = fetch_3c(iris, t_start, t_end)
        if waveform is None:
            print(f"{i:<4} {label:<30} {mag:<6.1f} {dist:>9.1f}°  no waveform")
            continue

        score = scorer.score(waveform)
        rows.append(("earthquake", label, score))
        print(f"{i:<4} {label:<30} {mag:<6.1f} {dist:>9.1f}°  score={score:.4f}")

    # ── Synthetic noise windows ───────────────────────────────────────────────
    # IRIS has patchy continuous data availability, so noise windows are
    # generated synthetically.  Two types are used:
    #   - white noise  (flat spectrum, hardest case for the classifier)
    #   - pink noise   (1/f spectrum, closer to real ambient seismic noise)
    print(f"\nScoring {N_NOISE_SAMPLES} synthetic noise waveforms...")
    print("-" * 70)

    rng = np.random.default_rng(seed=42)

    for i in range(N_NOISE_SAMPLES):
        if i % 2 == 0:
            # White noise normalised to unit std per channel
            raw = rng.standard_normal((N_SAMPLES, 3)).astype(np.float32)
            label = f"white_noise_{i:02d}"
        else:
            # Pink noise: whiten in frequency domain then apply 1/f colouring
            raw = np.zeros((N_SAMPLES, 3), dtype=np.float32)
            for ch in range(3):
                freqs = np.fft.rfftfreq(N_SAMPLES)
                freqs[0] = 1e-10   # avoid division by zero at DC
                spectrum = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
                spectrum /= np.sqrt(freqs)
                sig = np.fft.irfft(spectrum, n=N_SAMPLES).astype(np.float32)
                raw[:, ch] = sig / (sig.std() + 1e-10)
            label = f"pink_noise_{i:02d}"

        score = scorer.score(raw)
        rows.append(("noise", label, score))
        print(f"  {label}  score={score:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    eq_scores    = [r[2] for r in rows if r[0] == "earthquake"]
    noise_scores = [r[2] for r in rows if r[0] == "noise"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def stats(scores, label):
        if not scores:
            print(f"  {label}: no data")
            return
        a = np.array(scores)
        print(f"  {label} (n={len(a)}): "
              f"mean={a.mean():.4f}  std={a.std():.4f}  "
              f"min={a.min():.4f}  max={a.max():.4f}")

    stats(eq_scores,    "Earthquakes")
    stats(noise_scores, "Noise      ")

    if eq_scores and noise_scores:
        overlap = sum(s < 0.5 for s in eq_scores) + sum(s >= 0.5 for s in noise_scores)
        total   = len(eq_scores) + len(noise_scores)
        print(f"\n  Accuracy (eq≥0.5, noise<0.5): "
              f"{(total - overlap) / total * 100:.1f}%  "
              f"({total - overlap}/{total} correct)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "label", "score"])
        w.writerows(rows)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
