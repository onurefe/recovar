#!/usr/bin/env python3
"""
mseed_predictor.py — Score a 3-component MiniSEED file with RECOVAR.

For each overlapping window:
  1. A 40-second window is fetched from each component (ZNE or Z12).
  2. Windows that contain a gap on any component are dropped.
  3. Each component is resampled to 100 Hz if required.
  4. A 1–20 Hz ideal Fourier bandpass is applied per component.
  5. The inner 30 seconds (crop 5 s from each edge) are extracted.
  6. Each channel is demeaned then L2-normalised along the time axis
     (matches the preprocessing in the training data generator).
  7. The classifier returns an earthquake probability in [0, 1].

Output: a table of window start times and scores, optionally saved as CSV.

Usage:
    python mseed_predictor.py \\
        --input   waveforms.mseed \\
        --model   models/representation_cross_covariances.h5 \\
        [--step   10]           # step between windows in seconds (default: 10)
        [--output scores.csv]   # omit to print to stdout
"""

import argparse
import csv
import sys
import os
import numpy as np
from scipy.signal import resample as sp_resample
from obspy import read, Stream

# ---------------------------------------------------------------------------
# Constants — must match the trained model
# ---------------------------------------------------------------------------

TARGET_FS   = 100            # Hz
WINDOW_S    = 40.0           # total fetched window (filter buffer included)
CROP_EDGE_S = 5.0            # seconds stripped from each edge after filtering
INNER_S     = WINDOW_S - 2 * CROP_EDGE_S    # 30 s → 3000 samples for the model
BP_LOW_HZ   = 1.0
BP_HIGH_HZ  = 20.0

N_FETCH  = int(WINDOW_S    * TARGET_FS)   # 4000
N_INNER  = int(INNER_S     * TARGET_FS)   # 3000
CROP_OFF = int(CROP_EDGE_S * TARGET_FS)   # 500

# Component selection priority (name → channel suffixes to try in order)
COMP_MAP = [
    ("Z",  ["Z"]),
    ("H1", ["N", "1"]),
    ("H2", ["E", "2"]),
]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _resample(arr: np.ndarray, src_fs: float) -> np.ndarray:
    if abs(src_fs - TARGET_FS) < 0.01:
        return arr
    n_out = int(round(len(arr) * TARGET_FS / src_fs))
    return sp_resample(arr, n_out)


def _bandpass(arr: np.ndarray) -> np.ndarray:
    """Ideal 1–20 Hz rectangular Fourier-domain mask (zero-phase by construction)."""
    n     = len(arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_FS)
    mask  = (freqs >= BP_LOW_HZ) & (freqs <= BP_HIGH_HZ)
    return np.fft.irfft(np.fft.rfft(arr) * mask, n=n)


def _normalize(waveform: np.ndarray) -> np.ndarray:
    """Demean then L2-normalize each channel along the time axis.

    Matches DataGenerator._get_batchx() in reproducibility/data_generator.py:
        x = x - np.mean(x, axis=1, keepdims=True)
        x = x / (1e-37 + np.sqrt(np.sum(x**2, axis=1, keepdims=True)))

    Input shape: (N_INNER, 3)  — axis 0 is time, axis 1 is channel.
    """
    x    = waveform - waveform.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(x ** 2, axis=0, keepdims=True))
    return x / (1e-37 + norm)


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def preprocess_stream(st):
    """Return a copy of *st* resampled to TARGET_FS and bandpass-filtered.

    Applies the same resample → 1-20 Hz Fourier bandpass used per window
    inside score_file(), so the plot shows exactly what the model observes.
    Per-window demean and L2 normalisation are intentionally omitted here
    because they rescale each 30-second window independently and would
    produce discontinuous jumps in a continuous-data plot.
    """
    out = Stream()
    for tr in st:
        arr = _resample(tr.data.astype(np.float64), tr.stats.sampling_rate)
        arr = _bandpass(arr)
        tr2 = tr.copy()
        tr2.data = arr.astype(np.float32)
        tr2.stats.sampling_rate = TARGET_FS
        out.append(tr2)
    return out

def _select_trace(st, candidates: list):
    for suffix in candidates:
        matches = [tr for tr in st if tr.stats.channel.endswith(suffix)]
        if matches:
            return matches[0]
    return None


def _extract_raw(tr, t_start, n_samples: int) -> "np.ndarray | None":
    """
    Slice exactly n_samples from tr starting at t_start after resampling to
    TARGET_FS. Returns None if the slice contains a gap or is too short.
    """
    t_end  = t_start + n_samples / TARGET_FS
    sliced = tr.slice(t_start, t_end)
    if sliced is None or len(sliced) == 0:
        return None

    # Masked array means a gap was present within this window
    if np.ma.is_masked(sliced.data):
        return None

    arr = _resample(sliced.data.astype(np.float64), sliced.stats.sampling_rate)
    if len(arr) < n_samples:
        return None

    return arr[:n_samples]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_scorer(model_path: str):
    from recovar.representation_learning_models import RepresentationLearningMultipleAutoencoder
    from recovar.classifier_models import ClassifierMultipleAutoencoder

    dummy = np.zeros((1, N_INNER, 3), dtype=np.float32)
    model = RepresentationLearningMultipleAutoencoder(
        name="rep_learning_autoencoder_ensemble",
        input_noise_std=1e-6,
        eps=1e-27,
    )
    model.compile()
    model(dummy)
    model.load_weights(model_path)
    classifier = ClassifierMultipleAutoencoder(model)

    def score(waveform: np.ndarray) -> float:
        """Score a (3000, 3) float32 array. Returns earthquake probability [0, 1]."""
        return float(classifier(waveform[np.newaxis].astype(np.float32))[0])

    return score


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def score_file(mseed_path: str, model_path: str, step_s: float):
    """
    Yield (window_start, inner_start, score) for each gap-free window.

    window_start : start of the 40-second fetch window
    inner_start  : start of the scored 30-second window (window_start + 5 s)
    score        : earthquake probability [0, 1]
    """
    print(f"Loading {mseed_path} ...", file=sys.stderr)
    st = read(mseed_path)

    # Merge traces; gaps become masked array entries
    st.merge(method=0, fill_value=None)

    traces = {}
    for name, candidates in COMP_MAP:
        tr = _select_trace(st, candidates)
        if tr is None:
            available = [tr.stats.channel for tr in st]
            raise RuntimeError(
                f"Component '{name}' not found. "
                f"Available channels: {available}"
            )
        traces[name] = tr

    t_start = max(tr.stats.starttime for tr in traces.values())
    t_end   = min(tr.stats.endtime   for tr in traces.values())

    print(f"Stream spans {t_start} — {t_end}", file=sys.stderr)
    print(f"Loading model from {model_path} ...", file=sys.stderr)
    scorer = _load_scorer(model_path)
    print("Model ready. Scoring windows ...", file=sys.stderr)

    n_total   = 0
    n_dropped = 0
    t         = t_start

    while t + WINDOW_S <= t_end:
        n_total += 1
        channels = []
        gap_found = False

        for name in ("Z", "H1", "H2"):
            raw = _extract_raw(traces[name], t, N_FETCH)
            if raw is None:
                gap_found = True
                break
            filtered = _bandpass(raw)
            channels.append(filtered[CROP_OFF : CROP_OFF + N_INNER])

        if gap_found:
            n_dropped += 1
        else:
            waveform    = _normalize(np.stack(channels, axis=-1).astype(np.float32))
            inner_start = t + CROP_EDGE_S
            yield t, inner_start, scorer(waveform)

        t += step_s

    print(
        f"Done. {n_total} windows total, {n_dropped} dropped (gaps), "
        f"{n_total - n_dropped} scored.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="MiniSEED file path")
    parser.add_argument("--model",  "-m", required=True,
                        help="Model weights (.h5)")
    parser.add_argument("--step",   "-s", type=float, default=10.0,
                        help="Step between windows in seconds (default: 10)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()

    rows = list(score_file(args.input, args.model, args.step))

    header = ["window_start", "inner_start", "score"]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t_win, t_inner, score in rows:
                writer.writerow([str(t_win), str(t_inner), f"{score:.4f}"])
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    else:
        col = "{:<32}  {:<32}  {}"
        print(col.format(*header))
        print("-" * 74)
        for t_win, t_inner, score in rows:
            print(col.format(str(t_win), str(t_inner), f"{score:.4f}"))


if __name__ == "__main__":
    main()
