#!/usr/bin/env python3
"""
recovar_pick_filter.py — SeisComP module that attaches a recovar earthquake
probability score as a comment to every incoming Pick.

Usage (standalone):
    python recovar_pick_filter.py --model-path /path/to/weights.h5

Usage (as a SeisComP module):
    Copy to $SEISCOMP_ROOT/share/apps/ or $SEISCOMP_ROOT/bin/, then:
    seiscomp exec recovar_pick_filter

Configuration (recovar_pick_filter.cfg):
    recovar.modelPath = /path/to/representation_cross_covariances.h5
    recordStream      = slink://localhost:18000
"""

import sys
import os
import numpy as np

import seiscomp.client
import seiscomp.core
import seiscomp.datamodel
import seiscomp.io
import seiscomp.logging

# Add the recovar repo root to sys.path so the package is importable when the
# module is run from outside the repo directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from recovar_scorer import RecovARScorer  # noqa: E402  (local import after path fixup)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_BEFORE_S = 5.0    # seconds before pick time
WINDOW_AFTER_S  = 25.0   # seconds after pick time  →  total 30 s = 3000 @ 100 Hz
TARGET_FS       = 100    # Hz required by recovar

COMMENT_KEY = "recovar_score"


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class RecovARPickFilter(seiscomp.client.Application):

    def __init__(self, argc, argv):
        super().__init__(argc, argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(True, True)
        self.setPrimaryMessagingGroup("PICK")
        self.addMessagingSubscription("PICK")

        self._scorer: RecovARScorer | None = None
        self._model_path: str | None = None
        self._record_stream_url: str | None = None

    # ------------------------------------------------------------------
    # SeisComP lifecycle
    # ------------------------------------------------------------------

    def createCommandLineDescription(self):
        self.commandline().addGroup("RecovAR")
        self.commandline().addStringOption(
            "RecovAR", "model-path,m",
            "Path to the recovar model weights (.h5)",
        )
        return True

    def initConfiguration(self):
        if not super().initConfiguration():
            return False
        try:
            self._model_path = self.configGetString("recovar.modelPath")
        except Exception:
            pass
        try:
            self._record_stream_url = self.configGetString("recordStream")
        except Exception:
            pass
        return True

    def init(self):
        if not super().init():
            return False

        # Command-line flag overrides the config file value.
        try:
            self._model_path = self.commandline().optionString("model-path")
        except Exception:
            pass

        if not self._model_path:
            seiscomp.logging.error(
                "recovar_pick_filter: no model path set. "
                "Use --model-path or set recovar.modelPath in the config."
            )
            return False

        if not self._record_stream_url:
            seiscomp.logging.error(
                "recovar_pick_filter: recordStream not configured."
            )
            return False

        seiscomp.logging.info(f"recovar: loading model from {self._model_path}")
        try:
            self._scorer = RecovARScorer(self._model_path)
        except Exception as exc:
            seiscomp.logging.error(f"recovar: failed to load model — {exc}")
            return False

        seiscomp.logging.info("recovar_pick_filter: ready")
        return True

    # ------------------------------------------------------------------
    # Pick handling — called by the SeisComP messaging layer
    # ------------------------------------------------------------------

    def addObject(self, parentID, obj):
        pick = seiscomp.datamodel.Pick.Cast(obj)
        if pick:
            self._process_pick(pick)

    def updateObject(self, parentID, obj):
        pick = seiscomp.datamodel.Pick.Cast(obj)
        if pick:
            self._process_pick(pick)

    # ------------------------------------------------------------------

    def _process_pick(self, pick):
        try:
            wid  = pick.waveformID()
            net  = wid.networkCode()
            sta  = wid.stationCode()
            loc  = wid.locationCode()
            cha  = wid.channelCode()
            t    = pick.time().value()

            seiscomp.logging.debug(
                f"recovar: scoring {pick.publicID()} "
                f"[{net}.{sta}.{loc}.{cha} @ {t.iso()}]"
            )

            waveform = self._fetch_waveform(net, sta, loc, cha, t)
            if waveform is None:
                seiscomp.logging.warning(
                    f"recovar: skipping {pick.publicID()} — waveform unavailable"
                )
                return

            score = self._scorer.score(waveform)
            seiscomp.logging.info(
                f"recovar: {pick.publicID()} {COMMENT_KEY}={score:.4f}"
            )
            self._attach_comment(pick, score)

        except Exception as exc:
            seiscomp.logging.error(
                f"recovar: error on {pick.publicID()} — {exc}"
            )

    # ------------------------------------------------------------------
    # Waveform retrieval
    # ------------------------------------------------------------------

    def _fetch_waveform(
        self,
        net: str,
        sta: str,
        loc: str,
        cha: str,
        pick_time,
    ) -> np.ndarray | None:
        """
        Fetch a 30-second, 3-component window around *pick_time*.

        Returns a float32 array of shape (3000, 3) with channels ordered
        [Z, N/1, E/2], or None if the data cannot be retrieved.
        """
        t_start = pick_time + seiscomp.core.TimeSpan(-WINDOW_BEFORE_S)
        t_end   = pick_time + seiscomp.core.TimeSpan(WINDOW_AFTER_S)
        n_want  = int((WINDOW_BEFORE_S + WINDOW_AFTER_S) * TARGET_FS)  # 3000

        band = cha[:2]  # e.g. "HH" from "HHZ"

        raw = self._stream_components(net, sta, loc, band, t_start, t_end)
        if not raw:
            return None

        z  = self._select(raw, ["Z"])
        h1 = self._select(raw, ["N", "1"])
        h2 = self._select(raw, ["E", "2"])

        missing = [name for name, data in [("Z", z), ("H1", h1), ("H2", h2)]
                   if data is None]
        if missing:
            seiscomp.logging.warning(
                f"recovar: missing components {missing} "
                f"for {net}.{sta}.{loc}.{band}"
            )
            return None

        channels = [self._fit(arr, n_want) for arr in (z, h1, h2)]
        return np.stack(channels, axis=-1).astype(np.float32)  # (3000, 3)

    def _stream_components(self, net, sta, loc, band, t_start, t_end):
        """
        Open the configured RecordStream, request all components for the
        given band+wildcard, and return a dict mapping the last channel
        character to a resampled numpy array.
        """
        rs = seiscomp.io.RecordStream.Open(self._record_stream_url)
        if rs is None:
            seiscomp.logging.error(
                f"recovar: cannot open record stream {self._record_stream_url}"
            )
            return {}

        for comp in ("Z", "N", "E", "1", "2"):
            rs.addStream(net, sta, loc, band + comp, t_start, t_end)

        ri = seiscomp.io.RecordInput(
            rs,
            seiscomp.core.Array.DOUBLE,
            seiscomp.core.Record.SAVE_RAW,
        )

        buffers: dict[str, list[np.ndarray]] = {}
        for rec in ri:
            comp_char = rec.channelCode()[-1]
            arr = _record_to_numpy(rec)
            arr = _resample(arr, rec.samplingFrequency(), TARGET_FS)
            buffers.setdefault(comp_char, []).append(arr)

        rs.close()
        return {c: np.concatenate(segs) for c, segs in buffers.items()}

    @staticmethod
    def _select(data: dict, candidates: list[str]) -> np.ndarray | None:
        for c in candidates:
            if c in data:
                return data[c]
        return None

    @staticmethod
    def _fit(arr: np.ndarray, n: int) -> np.ndarray:
        """Trim or zero-pad *arr* to exactly *n* samples."""
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

    # ------------------------------------------------------------------
    # Attach comment via SeisComP messaging
    # ------------------------------------------------------------------

    def _attach_comment(self, pick, score: float):
        ci = seiscomp.datamodel.CreationInfo()
        ci.setAgencyID(self.agencyID())
        ci.setAuthor("recovar_pick_filter")
        ci.setCreationTime(seiscomp.core.Time.GMT())

        comment = seiscomp.datamodel.Comment()
        comment.setText(f"{COMMENT_KEY}:{score:.4f}")
        comment.setCreationInfo(ci)

        seiscomp.datamodel.Notifier.Enable()
        pick.add(comment)
        msg = seiscomp.datamodel.Notifier.GetMessage()
        seiscomp.datamodel.Notifier.Disable()

        if msg and self.connection():
            self.connection().send("PICK", msg)


# ---------------------------------------------------------------------------
# Helpers (module-level, used by _stream_components)
# ---------------------------------------------------------------------------

def _record_to_numpy(rec) -> np.ndarray:
    """Convert a SeisComP GenericRecord to a numpy float64 array."""
    data = rec.data()
    # Modern SeisComP (≥4.x) exposes a numpy() buffer on typed arrays.
    try:
        return np.frombuffer(data.numpy(), dtype=np.float64).copy()
    except (AttributeError, TypeError):
        # Fallback: iterate element by element (slower but universally safe).
        return np.array([data.get(i) for i in range(data.size())], dtype=np.float64)


def _resample(arr: np.ndarray, src_fs: float, dst_fs: int) -> np.ndarray:
    if abs(src_fs - dst_fs) < 0.01:
        return arr
    from scipy.signal import resample as sp_resample
    n_out = int(round(len(arr) * dst_fs / src_fs))
    return sp_resample(arr, n_out)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = RecovARPickFilter(len(sys.argv), sys.argv)
    sys.exit(app())
