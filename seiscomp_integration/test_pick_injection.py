#!/usr/bin/env python3
"""
test_pick_injection.py — injects a test Pick into SeisComP messaging and
verifies that recovar_pick_filter responds with a recovar_score comment.

Usage:
    python3 test_pick_injection.py [-T 60]

Requirements:
    - scmaster must be running
    - recovar_pick_filter must be running and ready
    - The pick filter's recordStream must cover IU.ANMO.00.HH? on 2019-07-06

Download test waveform data (covers the injected pick time):
    mkdir -p ~/seiscomp_test/data
    wget -O ~/seiscomp_test/data/IU.ANMO.HH.mseed \
      "https://service.iris.edu/fdsnws/dataselect/1/query?\
network=IU&station=ANMO&location=00&channel=HH%3F\
&starttime=2019-07-06T03:20:30&endtime=2019-07-06T03:23:00&format=miniseed"

Then set in ~/seiscomp/etc/recovar_pick_filter.cfg:
    recordStream = file:///home/<user>/seiscomp_test/data/IU.ANMO.HH.mseed
"""

import sys
import os
import time

import seiscomp.client
import seiscomp.core
import seiscomp.datamodel

PICK_ID       = f"recovar_test_pick_{int(time.time())}"   # unique per run
PICK_TIME_STR = "2019-07-06T03:22:00.000Z"
NETWORK       = "IU"
STATION       = "ANMO"
LOCATION      = "00"
CHANNEL       = "HHZ"
LOG_FILE      = os.path.expanduser("~/.seiscomp/log/recovar_pick_filter.log")


class PickInjector(seiscomp.client.Application):

    def __init__(self, argc, argv):
        super().__init__(argc, argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(False, False)
        self.setPrimaryMessagingGroup("PICK")
        self._wait = 60

    def createCommandLineDescription(self):
        self.commandline().addGroup("Test")
        self.commandline().addIntOption("Test", "wait,T",
            "Seconds to wait for a scored pick response (default: 60)")
        return True

    def init(self):
        if not super().init():
            return False
        try:
            self._wait = self.commandline().optionInt("wait")
        except Exception:
            pass
        return True

    def run(self):
        pick = seiscomp.datamodel.Pick.Create(PICK_ID)

        wid = seiscomp.datamodel.WaveformStreamID()
        wid.setNetworkCode(NETWORK)
        wid.setStationCode(STATION)
        wid.setLocationCode(LOCATION)
        wid.setChannelCode(CHANNEL)
        pick.setWaveformID(wid)

        t = seiscomp.core.Time()
        t.fromString(PICK_TIME_STR, "%FT%T.%fZ")
        tq = seiscomp.datamodel.TimeQuantity()
        tq.setValue(t)
        pick.setTime(tq)
        pick.setEvaluationMode(seiscomp.datamodel.AUTOMATIC)

        ci = seiscomp.datamodel.CreationInfo()
        ci.setAgencyID("TEST")
        ci.setAuthor("test_pick_injection")
        ci.setCreationTime(seiscomp.core.Time.GMT())
        pick.setCreationInfo(ci)

        seiscomp.datamodel.Notifier.Enable()
        ep = seiscomp.datamodel.EventParameters()
        ep.add(pick)
        msg = seiscomp.datamodel.Notifier.GetMessage(True)
        seiscomp.datamodel.Notifier.Disable()

        if not self.connection().send("PICK", msg):
            print("FAIL: could not send pick to messaging")
            return False

        # Record log file size before sending — only look at lines appended after this
        log_offset = _log_size()

        print(f"Sent pick {PICK_ID}")
        print(f"  station : {NETWORK}.{STATION}.{LOCATION}.{CHANNEL}")
        print(f"  time    : {PICK_TIME_STR}")
        print(f"Waiting up to {self._wait}s for recovar_score in log...")

        # Poll the pick filter log for the score
        deadline = time.time() + self._wait
        while time.time() < deadline:
            time.sleep(1)
            score = _find_score_in_log(PICK_ID, offset=log_offset)
            if score is not None:
                print(f"\nresult : recovar_score:{score:.4f}")
                if score >= 0.5:
                    print(f"PASS   — score {score:.4f} >= 0.5 (high-confidence seismic signal)")
                else:
                    print(f"WARN   — score {score:.4f} < 0.5 (low confidence; check waveform window)")
                return True

        print(f"\nFAIL   — no recovar_score found in log within {self._wait}s")
        print(f"         Is recovar_pick_filter running and ready?")
        print(f"         Does the recordStream cover IU.ANMO.00.HH? on 2019-07-06?")
        return False


def _log_size() -> int:
    """Return current byte size of the log file (0 if missing)."""
    try:
        return os.path.getsize(LOG_FILE)
    except FileNotFoundError:
        return 0


def _find_score_in_log(pick_id: str, offset: int = 0):
    """Return the score float if pick_id appears in new log lines after `offset` bytes."""
    try:
        with open(LOG_FILE) as f:
            f.seek(offset)
            for line in f:
                if pick_id in line and "recovar_score=" in line:
                    return float(line.split("recovar_score=")[1].strip())
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


if __name__ == "__main__":
    app = PickInjector(len(sys.argv), sys.argv)
    ok = app()
    # ok is 0 (success) or 1 (failure) from Application.__call__
    # also check that run() returned True
    sys.exit(ok)
