#!/usr/bin/env python3
"""
create_test_mseed.py — Download a short waveform from IRIS and save as MiniSEED.

Downloads ~450 s of KO.DKL..HH? centred on the P arrival of a known
earthquake, suitable for testing mseed_predictor.py.

KO.DKL (Dikili, western Turkey) is ~100 km from the Balıkesir epicentres,
putting the events in the regional distance range where the 1–20 Hz band
used by RECOVAR carries strong P- and S-wave energy.

Default event catalog focuses on Balıkesir and the broader Marmara region.

Usage:
    python create_test_mseed.py
    python create_test_mseed.py --output data/test.mseed --event 0
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

NET, STA, LOC, BAND = "KO", "DKL", "", "HH"
STA_LAT, STA_LON   = 39.0713, 26.9053   # Dikili, western Turkey

EVENTS = [
    ("2025-08-10T16:53:47", 39.312, 28.069,  9.4, "M6.1 Bigadic Balikesir 2025"),
    ("2025-10-27T19:48:29", 39.233, 28.228,  8.0, "M6.0 Sindirgi Balikesir 2025"),
    ("2025-04-23T09:49:10", 40.840, 28.142, 12.7, "M6.2 Sea of Marmara 2025"),
    ("2023-02-06T01:17:35", 37.174, 37.032, 10.0, "M7.8 Kahramanmaras Turkey 2023"),
]

PRE_P_S  = 60.0   # seconds of pre-event noise before P arrival
POST_P_S = 390.0  # seconds after P arrival (total window ≈ 450 s)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", default="data/test.mseed",
                        help="Output MiniSEED path (default: data/test.mseed)")
    parser.add_argument("--event", "-e", type=int, default=0,
                        choices=range(len(EVENTS)),
                        help="Event index (default: 0 = Bigadic Balikesir 2025)")
    args = parser.parse_args()

    origin_str, lat, lon, dep, label = EVENTS[args.event]
    origin = UTCDateTime(origin_str)

    print(f"Event   : {label}")
    print(f"Station : {NET}.{STA}.{LOC}.{BAND}?")

    taup = TauPyModel("iasp91")
    dist = locations2degrees(STA_LAT, STA_LON, lat, lon)
    arrs = taup.get_travel_times(dep, dist, phase_list=["P", "p", "Pn", "Pg"])
    if not arrs:
        raise RuntimeError("No P arrival found for this event / station pair.")

    p_time  = origin + arrs[0].time
    t_start = p_time - PRE_P_S
    t_end   = p_time + POST_P_S

    print(f"P arrival : {p_time}  (dist={dist:.1f}°)")
    print(f"Window    : {t_start}  →  {t_end}  ({PRE_P_S + POST_P_S:.0f} s)")
    print("Downloading from IRIS ...")

    client = Client("IRIS")
    st = client.get_waveforms(NET, STA, LOC, BAND + "?", t_start, t_end)
    if not st:
        raise RuntimeError("No data returned from IRIS.")

    st.write(args.output, format="MSEED")

    import os
    size_kb = os.path.getsize(args.output) / 1024
    duration = st[0].stats.endtime - st[0].stats.starttime
    print(f"Written   : {args.output}  ({size_kb:.0f} KB, {duration:.0f} s, {len(st)} traces)")


if __name__ == "__main__":
    main()
