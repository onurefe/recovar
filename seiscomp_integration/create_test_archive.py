#!/usr/bin/env python3
"""
create_test_archive.py — Download waveforms from IRIS and write them into
an SDS archive compatible with SeisComP's sdsarchive:// record stream.

SDS layout:
  <root>/<year>/<net>/<sta>/<cha.D>/<net>.<sta>.<loc>.<cha>.D.<year>.<doy>

Usage:
    python3 create_test_archive.py
    python3 create_test_archive.py --output ~/seiscomp_test/sds --duration 300
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

NET, STA, LOC, BAND = "IU", "ANMO", "00", "HH"
STA_LAT, STA_LON   = 34.9459, -106.4572

# Earthquake events confirmed to have IRIS waveform data at IU.ANMO.00.HH?.
# (origin time, lat, lon, depth_km, label)
EVENTS = [
    ("2019-07-06T03:19:53", 35.770, -117.599,  8.0, "M7.1 Ridgecrest 2019"),
    ("2021-07-29T06:15:49", 55.364, -157.888, 35.0, "M8.2 Alaska 2021"),
    ("2018-08-19T00:19:38", 56.046, -149.073, 14.2, "M8.2 Alaska 2018"),
    ("2020-07-22T06:12:44", 55.070,  158.596, 10.0, "M7.8 Russia 2020"),
    ("2020-01-28T19:10:23", 17.860,  -66.050, 10.0, "M6.4 Puerto Rico 2020"),
    ("2022-09-19T18:05:07", 18.330,  -99.750, 15.0, "M7.6 Mexico 2022"),
    ("2023-05-10T02:50:35", -17.96, -178.10,  10.0, "M7.6 Tonga 2023"),
    ("2018-09-06T15:49:12",  7.484,  -34.786, 10.0, "M7.9 Atlantic 2018"),
]

# Quiet periods at times when no large event is expected at ANMO.
NOISE_WINDOWS = [
    ("2021-03-01T00:00:00", "noise_2021-03-01"),
    ("2019-02-05T06:00:00", "noise_2019-02-05"),
    ("2021-02-24T12:00:00", "noise_2021-02-24"),
    ("2023-09-12T18:00:00", "noise_2023-09-12"),
    ("2024-01-10T09:00:00", "noise_2024-01-10"),
]

WINDOW_BEFORE = 30.0   # s before P arrival (extra buffer for scautopick)
WINDOW_AFTER  = 120.0  # s after  P arrival


def pad_to_day(tr):
    """Pad a trace with zeros to cover its full calendar day (midnight-to-midnight).

    Full-day coverage is required so SeisComP's sdsarchive record stream
    can serve the file without reporting an invalid time window.

    We build the padded array manually rather than using obspy trim so that
    starttime is set to exactly UTC midnight.  obspy's trim(nearest_sample=True)
    snaps to the nearest sample on the existing grid, which often lands one
    sample before midnight and shifts data into the previous calendar day.
    """
    t   = tr.stats.starttime
    fs  = tr.stats.sampling_rate
    day_start = UTCDateTime(t.year, t.month, t.day)
    n_pre     = int(round((t - day_start) * fs))
    n_day     = int(round(86400 * fs))
    n_post    = max(0, n_day - n_pre - len(tr.data))

    tr              = tr.copy()
    tr.data         = np.concatenate([
        np.zeros(n_pre, dtype=np.int32),
        tr.data.astype(np.int32),
        np.zeros(n_post, dtype=np.int32),
    ])
    tr.stats.starttime = day_start
    return tr


def sds_path(root, tr):
    """Return the SDS file path for a given trace."""
    t    = tr.stats.starttime
    net  = tr.stats.network
    sta  = tr.stats.station
    loc  = tr.stats.location
    cha  = tr.stats.channel
    year = t.year
    doy  = t.julday
    dir_ = os.path.join(root, str(year), net, sta, f"{cha}.D")
    fname = f"{net}.{sta}.{loc}.{cha}.D.{year}.{doy:03d}"
    return os.path.join(dir_, fname)


def write_sds(root, st):
    """Write all traces in *st* to their SDS files, padded to full-day coverage."""
    written = set()
    for tr in st:
        path = sds_path(root, tr)   # use original starttime for correct day name
        tr   = pad_to_day(tr)       # padding shifts starttime to 23:59:59 of prev day
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            existing = read_mseed_safe(path)
            if existing is not None:
                existing += tr.copy()
                existing.merge(method=1, fill_value=0)
                for ex_tr in existing:
                    ex_tr = pad_to_day(ex_tr)
                    ex_tr.write(path, format="MSEED", reclen=512, encoding="STEIM2")
            else:
                tr.write(path, format="MSEED", reclen=512, encoding="STEIM2")
        else:
            tr.write(path, format="MSEED", reclen=512, encoding="STEIM2")
        written.add(path)
    return written


def read_mseed_safe(path):
    try:
        from obspy import read
        return read(path)
    except Exception:
        return None


def fetch(client, t_start, t_end):
    try:
        st = client.get_waveforms(NET, STA, LOC, BAND + "?", t_start, t_end)
        return st if len(st) > 0 else None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Build a SeisComP SDS test archive.")
    p.add_argument("--output",   default=os.path.expanduser("~/seiscomp_test/sds"),
                   help="SDS root directory (default: ~/seiscomp_test/sds)")
    p.add_argument("--duration", type=float, default=WINDOW_BEFORE + WINDOW_AFTER,
                   help="Window length in seconds around each event (default: 150)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    iris = Client("IRIS")
    usgs = Client("USGS")
    taup = TauPyModel("iasp91")

    total_written = []

    # ── Earthquake windows ────────────────────────────────────────────────────
    print("=== Earthquake windows ===")
    for origin_str, lat, lon, dep, label in EVENTS:
        origin = UTCDateTime(origin_str)
        dist   = locations2degrees(STA_LAT, STA_LON, lat, lon)
        arrs   = taup.get_travel_times(dep, dist, phase_list=["P", "p"])
        if not arrs:
            print(f"  SKIP  {label} — no P arrival")
            continue
        p_time  = origin + arrs[0].time
        t_start = p_time  - WINDOW_BEFORE
        t_end   = p_time  + (args.duration - WINDOW_BEFORE)

        st = fetch(iris, t_start, t_end)
        if st is None:
            print(f"  SKIP  {label} — no waveform at IRIS")
            continue

        written = write_sds(args.output, st)
        total_written.extend(written)
        print(f"  OK    {label}  [{t_start}  dist={dist:.1f}°]  → {len(written)} file(s)")

    # ── Noise windows ─────────────────────────────────────────────────────────
    print("\n=== Noise windows ===")
    for t_str, label in NOISE_WINDOWS:
        t_start = UTCDateTime(t_str)
        t_end   = t_start + args.duration

        st = fetch(iris, t_start, t_end)
        if st is None:
            print(f"  SKIP  {label} — no waveform at IRIS")
            continue

        written = write_sds(args.output, st)
        total_written.extend(written)
        print(f"  OK    {label}  → {len(written)} file(s)")

    # ── Also convert the existing local Ridgecrest file ───────────────────────
    local = os.path.expanduser("~/seiscomp_test/data/IU.ANMO.HH.mseed")
    if os.path.exists(local):
        print(f"\n=== Local file: {local} ===")
        from obspy import read
        st = read(local)
        written = write_sds(args.output, st)
        total_written.extend(written)
        print(f"  OK    IU.ANMO.HH.mseed  → {len(written)} file(s)")

    print(f"\nDone. {len(set(total_written))} SDS file(s) written to {args.output}")
    print(f"\nrecordStream URL for SeisComP config:")
    print(f"  sdsarchive://{args.output}")


if __name__ == "__main__":
    main()
