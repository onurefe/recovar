#!/usr/bin/env python3
"""
query_scored_picks.py — Print, save, and plot picks that received recovar_score
comments from recovar_pick_filter.

Usage:
    python3 query_scored_picks.py                              # print table
    python3 query_scored_picks.py -o picks.csv                 # print + CSV
    python3 query_scored_picks.py --plot --plot-output fig.png # waveform grid
"""

import argparse
import csv
import math
import os
import sys
import pymysql

DB = dict(host="localhost", user="sysop", password="sysop", database="seiscomp")

# recovar_score_sweep format:  "<key>:<start_s>:<step_s>:<s0>,<s1>,..."
SWEEP_KEY = "recovar_score_sweep"

QUERY = """
SELECT
    po.publicID                                             AS pick_id,
    CONCAT(p.time_value,'.',LPAD(p.time_value_ms,3,'0'))   AS pick_time,
    p.waveformID_networkCode                                AS net,
    p.waveformID_stationCode                                AS sta,
    p.waveformID_channelCode                                AS cha,
    CAST(REPLACE(c1.text,'recovar_score:','') AS DECIMAL(6,4)) AS score,
    c2.text                                                 AS sweep
FROM Pick p
JOIN  PublicObject po ON po._oid = p._oid
JOIN  Comment c1 ON c1._parent_oid = p._oid
                 AND c1.text LIKE 'recovar_score:%%'
                 AND c1.text NOT LIKE 'recovar_score_sweep:%%'
LEFT JOIN Comment c2 ON c2._parent_oid = p._oid
                     AND c2.text LIKE 'recovar_score_sweep:%%'
WHERE p.creationInfo_author LIKE 'scautopick%%'
ORDER BY p.time_value
"""

# Waveform window shown in the top panel (seconds before/after pick)
PLOT_BEFORE_S = 15.0
PLOT_AFTER_S  = 45.0


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def parse_sweep(text):
    """Parse 'recovar_score_sweep:<start>:<step>:<v0>,<v1>,...'
    Returns (offsets_list, scores_list) or (None, None) on failure.
    """
    if not text or not text.startswith(SWEEP_KEY):
        return None, None
    try:
        _, start_s, step_s, vals_str = text.split(":", 3)
        start  = int(start_s)
        step   = int(step_s)
        scores = [float(v) for v in vals_str.split(",")]
        offsets = [start + i * step for i in range(len(scores))]
        return offsets, scores
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Representative pick selection
# ---------------------------------------------------------------------------

def select_representatives(rows, max_panels=9):
    """One pick per event day: prefer picks with sweep data, then highest score."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        day = str(row[1])[:10]   # "YYYY-MM-DD"
        groups[day].append(row)
    reps = []
    for day in sorted(groups):
        with_sweep    = [r for r in groups[day] if r[6]]
        without_sweep = [r for r in groups[day] if not r[6]]
        candidates = with_sweep if with_sweep else without_sweep
        # publicID starts with Pick/YYYYMMDDHHMMSS so lexicographic max = most recent run
        best = max(candidates, key=lambda r: r[0])
        reps.append(best)
    return reps[:max_panels]


# ---------------------------------------------------------------------------
# Waveform fetch from SDS (for the top panel)
# ---------------------------------------------------------------------------

def fetch_waveform(sds_root, net, sta, loc, cha_prefix, pick_time_utc):
    """Read a short window from the SDS archive and return (times_rel, data).

    times_rel: seconds relative to pick. Returns (None, None) on failure.
    """
    import numpy as np
    from obspy import UTCDateTime
    from obspy.clients.filesystem.sds import Client as SDSClient

    t0 = UTCDateTime(str(pick_time_utc))
    try:
        client = SDSClient(sds_root)
        st = client.get_waveforms(net, sta, loc, cha_prefix + "Z",
                                  t0 - PLOT_BEFORE_S, t0 + PLOT_AFTER_S)
        if not st:
            return None, None
        tr = st.merge(fill_value=0)[0]
        tr.detrend("demean")
        fs    = tr.stats.sampling_rate
        arr   = tr.data.astype(float)
        freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
        mask  = (freqs >= 1.0) & (freqs <= 20.0)
        data  = np.fft.irfft(np.fft.rfft(arr) * mask, n=len(arr))
        times = tr.times() - PLOT_BEFORE_S
        return times, data
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_picks(reps, sds_root, output_path=None):
    import matplotlib
    if output_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n     = len(reps)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    # Two subplot rows per event if any pick has sweep data, one otherwise.
    has_sweep = any(row[6] for row in reps)
    subplot_rows = 2 if has_sweep else 1

    fig, axes = plt.subplots(
        nrows * subplot_rows, ncols,
        figsize=(6 * ncols, (2.8 if has_sweep else 3.2) * nrows * subplot_rows),
        squeeze=False,
    )
    fig.suptitle("RECOVAR-scored picks  (IU.ANMO.00.HHZ, 1–20 Hz bandpass)",
                 fontsize=11, y=1.01)

    for idx, (pick_id, pick_time, net, sta, cha, score, sweep_text) in enumerate(reps):
        col      = idx % ncols
        wf_row   = (idx // ncols) * subplot_rows
        sw_row   = wf_row + 1 if has_sweep else None

        ax_wf = axes[wf_row][col]
        score = float(score)
        color = "green" if score >= 0.5 else "red"
        band  = cha[:2]

        # ---- waveform panel ----
        times, data = fetch_waveform(sds_root, net, sta, "00", band, pick_time)
        if times is not None:
            ax_wf.plot(times, data, color="steelblue", lw=0.6)
            ax_wf.axvline(0, color="crimson", lw=1.2, ls="--")
            ax_wf.set_xlim(-PLOT_BEFORE_S, PLOT_AFTER_S)
        else:
            ax_wf.text(0.5, 0.5, "waveform unavailable",
                       ha="center", va="center", transform=ax_wf.transAxes,
                       color="grey", fontsize=8)
        ax_wf.set_title(f"{str(pick_time)[:19]}  |  score={score:.4f}",
                        fontsize=8.5, color=color)
        ax_wf.set_ylabel("counts", fontsize=7)
        ax_wf.tick_params(labelsize=7)
        if sw_row is None:
            ax_wf.set_xlabel("time rel. to pick (s)", fontsize=7)

        # ---- sweep panel ----
        if sw_row is not None:
            ax_sw = axes[sw_row][col]
            offsets, scores = parse_sweep(sweep_text)

            if offsets is not None:
                offsets = np.array(offsets, dtype=float)
                scores  = np.array(scores,  dtype=float)

                # Colour segments: green ≥ 0.5, red < 0.5
                for i in range(len(offsets) - 1):
                    seg_color = "green" if scores[i] >= 0.5 else "tomato"
                    ax_sw.plot(offsets[i:i+2], scores[i:i+2],
                               color=seg_color, lw=1.8)
                ax_sw.scatter(offsets, scores, s=18, zorder=3,
                              c=["green" if s >= 0.5 else "tomato" for s in scores])

                # Highlight t_p − 30 s
                if -30 in SWEEP_OFFSETS_S_REF:
                    hi_idx = SWEEP_OFFSETS_S_REF.index(-30)
                    if hi_idx < len(scores):
                        ax_sw.scatter([-30], [scores[hi_idx]], s=60,
                                      color="navy", zorder=5)
                        ax_sw.annotate(
                            f"t_p−30 s\n{scores[hi_idx]:.3f}",
                            xy=(-30, scores[hi_idx]),
                            xytext=(-30 + 4, scores[hi_idx] + 0.08),
                            fontsize=6.5, color="navy",
                            arrowprops=dict(arrowstyle="->", color="navy", lw=0.8),
                        )

                ax_sw.axhline(0.5, color="grey", ls="--", lw=0.8)
                ax_sw.axvline(0,   color="crimson", ls="--", lw=1.0)
                ax_sw.set_ylim(-0.05, 1.05)
                ax_sw.set_xlim(offsets[0] - 3, offsets[-1] + 3)
            else:
                ax_sw.text(0.5, 0.5, "no sweep data",
                           ha="center", va="center", transform=ax_sw.transAxes,
                           color="grey", fontsize=8)

            ax_sw.set_xlabel("window centre offset from t_p (s)", fontsize=7)
            ax_sw.set_ylabel("recovar score", fontsize=7)
            ax_sw.tick_params(labelsize=7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        col = idx % ncols
        for r in range(subplot_rows):
            axes[(idx // ncols) * subplot_rows + r][col].set_visible(False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


# Reference list matching SWEEP_OFFSETS_S in recovar_pick_filter.py
SWEEP_OFFSETS_S_REF = list(range(-40, 31, 5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Query and export recovar-scored picks.")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="Write results to a CSV file")
    ap.add_argument("--no-print", action="store_true",
                    help="Suppress terminal table output")
    ap.add_argument("--plot", action="store_true",
                    help="Plot representative waveforms (+ sweep if available)")
    ap.add_argument("--plot-output", metavar="FILE",
                    help="Save plot to file instead of displaying it")
    ap.add_argument("--sds", metavar="DIR",
                    default=os.path.expanduser("~/seiscomp_test/sds"),
                    help="SDS archive root (default: ~/seiscomp_test/sds)")
    args = ap.parse_args()

    try:
        conn = pymysql.connect(**DB)
    except pymysql.Error as e:
        sys.exit(f"DB connection failed: {e}")

    with conn:
        cur = conn.cursor()
        cur.execute(QUERY)
        rows = cur.fetchall()

    if not rows:
        print("No scored picks from scautopick found in database.")
        return

    if not args.no_print:
        header = f"{'pick_time':<26} {'net':<4} {'sta':<6} {'cha':<4} {'score':<6}  pick_id"
        print(header)
        print("-" * (len(header) - 2))
        for pick_id, pick_time, net, sta, cha, score, sweep in rows:
            sweep_flag = " [sweep]" if sweep else ""
            print(f"{str(pick_time):<26} {net:<4} {sta:<6} {cha:<4} "
                  f"{float(score):<6.4f}  {pick_id}{sweep_flag}")
        print(f"\n{len(rows)} scored pick(s) from scautopick.")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pick_time", "net", "sta", "cha", "score", "pick_id", "sweep"])
            for pick_id, pick_time, net, sta, cha, score, sweep in rows:
                w.writerow([pick_time, net, sta, cha, float(score), pick_id,
                            sweep or ""])
        print(f"Wrote {len(rows)} row(s) to {args.output}")

    if args.plot or args.plot_output:
        reps = select_representatives(rows)
        plot_picks(reps, args.sds, output_path=args.plot_output)


if __name__ == "__main__":
    main()
