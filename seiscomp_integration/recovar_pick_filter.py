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
import numpy as np

import seiscomp.client
import seiscomp.core
import seiscomp.datamodel
import seiscomp.io
import seiscomp.logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Waveform fetch window for the single pick score (filter edge buffer)
FETCH_BEFORE_S  = 20.0
FETCH_AFTER_S   = 20.0
CROP_BEFORE_S   = 15.0   # inner 30 s passed to RecovAR
CROP_AFTER_S    = 15.0

# Score sweep: wider fetch so every window fits; offsets relative to t_p
SWEEP_FETCH_BEFORE_S = 60.0
SWEEP_FETCH_AFTER_S  = 50.0
SWEEP_OFFSETS_S      = list(range(-40, 31, 5))  # −40,−35,…,0,…,+30 (15 pts)

TARGET_FS   = 100    # Hz required by recovar
BP_LOW_HZ   = 1.0
BP_HIGH_HZ  = 20.0
BP_CORNERS  = 4

COMMENT_KEY       = "recovar_score"
SWEEP_COMMENT_KEY = "recovar_score_sweep"


from recovar.representation_learning_models import RepresentationLearningMultipleAutoencoder
from recovar.classifier_models import ClassifierMultipleAutoencoder

_DUMMY = np.zeros((1, 3000, 3), dtype=np.float32)


class RecovARScorer:
    def __init__(self, model_path: str):
        self._model = RepresentationLearningMultipleAutoencoder(
            name="rep_learning_autoencoder_ensemble",
            input_noise_std=1e-6,
            eps=1e-27,
        )
        self._model.compile()
        self._model(_DUMMY)
        self._model.load_weights(model_path)
        self._classifier = ClassifierMultipleAutoencoder(self._model)

    def score(self, waveform: np.ndarray) -> float:
        """Score a (3000, 3) float32 waveform. Returns [0, 1]."""
        x = waveform[np.newaxis].astype(np.float32)
        return float(self._classifier(x)[0])


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class RecovARPickFilter(seiscomp.client.Application):

    def __init__(self, argc, argv):
        super().__init__(argc, argv)
        self.setMessagingEnabled(True)
        self.setDatabaseEnabled(False, False)
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
        self.commandline().addStringOption(
            "RecovAR", "record-stream,I",
            "RecordStream URL (e.g. slink://localhost:18000)",
        )
        return True

    def initConfiguration(self):
        if not super().initConfiguration():
            return False
        try:
            self._record_stream_url = self.configGetString("recordStream")
        except Exception:
            pass
        return True

    def init(self):
        if not super().init():
            return False

        try:
            self._model_path = self.commandline().optionString("model-path")
        except Exception:
            pass

        try:
            self._record_stream_url = self.commandline().optionString("record-stream")
        except Exception:
            pass

        if not self._model_path:
            seiscomp.logging.error("recovar_pick_filter: --model-path is required.")
            return False

        if not self._record_stream_url:
            seiscomp.logging.error(
                "recovar_pick_filter: --record-stream is required "
                "(or set recordStream in global.cfg)."
            )
            return False

        seiscomp.logging.notice(f"recovar: loading model from {self._model_path}")
        try:
            self._scorer = RecovARScorer(self._model_path)
        except Exception as exc:
            seiscomp.logging.error(f"recovar: failed to load model — {exc}")
            return False

        seiscomp.logging.notice("recovar_pick_filter: ready")
        return True

    # ------------------------------------------------------------------
    # Pick handling
    # ------------------------------------------------------------------

    def addObject(self, parentID, obj):
        pick = seiscomp.datamodel.Pick.Cast(obj)
        if pick:
            self._process_pick(pick)

    def updateObject(self, parentID, obj):
        pick = seiscomp.datamodel.Pick.Cast(obj)
        if pick:
            self._process_pick(pick)

    def _process_pick(self, pick):
        try:
            wid = pick.waveformID()
            net = wid.networkCode()
            sta = wid.stationCode()
            loc = wid.locationCode()
            cha = wid.channelCode()
            t   = pick.time().value()

            seiscomp.logging.debug(
                f"recovar: scoring {pick.publicID()} "
                f"[{net}.{sta}.{loc}.{cha} @ {t.iso()}]"
            )

            # One wide fetch of raw (unfiltered) data for both the pick score
            # and the sweep.  Bandpass is applied per-window inside _score_window
            # so that signal from outside each window cannot contaminate it.
            raw_data, pick_idx = self._fetch_waveform_raw(
                net, sta, loc, cha, t,
                before_s=SWEEP_FETCH_BEFORE_S,
                after_s=SWEEP_FETCH_AFTER_S,
                apply_bandpass=False,
            )
            if raw_data is None:
                seiscomp.logging.warning(
                    f"recovar: skipping {pick.publicID()} — waveform unavailable"
                )
                return

            # Score at t_p using independent per-window bandpass
            waveform = self._score_window(raw_data, pick_idx, 0.0)
            if waveform is None:
                seiscomp.logging.warning(
                    f"recovar: skipping {pick.publicID()} — score window at t_p failed"
                )
                return

            score = self._scorer.score(waveform)
            seiscomp.logging.notice(
                f"recovar: {pick.publicID()} {COMMENT_KEY}={score:.4f}"
            )
            self._attach_comment(pick, score)

            # Sweep: each offset gets its own isolated 40 s → bandpass → 30 s window
            sweep_scores = []
            for off in SWEEP_OFFSETS_S:
                w = self._score_window(raw_data, pick_idx, off)
                sweep_scores.append(
                    self._scorer.score(w) if w is not None else float("nan")
                )
            self._attach_sweep_comment(pick, sweep_scores)

        except Exception as exc:
            seiscomp.logging.error(
                f"recovar: error on {pick.publicID()} — {exc}"
            )

    # ------------------------------------------------------------------
    # Waveform retrieval
    # ------------------------------------------------------------------

    def _fetch_waveform_raw(
        self,
        net: str,
        sta: str,
        loc: str,
        cha: str,
        pick_time,
        before_s: float = FETCH_BEFORE_S,
        after_s: float = FETCH_AFTER_S,
        apply_bandpass: bool = True,
    ) -> tuple:
        """Fetch before_s+after_s seconds, optionally bandpass, return (data, pick_idx).

        data: float32 array (n_total, 3) ordered [Z, N/1, E/2].
        pick_idx: sample index of t_p within data.
        Returns (None, 0) on failure.
        """
        t_start  = pick_time + seiscomp.core.TimeSpan(-before_s)
        t_end    = pick_time + seiscomp.core.TimeSpan(after_s)
        n_total  = int((before_s + after_s) * TARGET_FS)
        pick_idx = int(before_s * TARGET_FS)
        band     = cha[:2]

        raw = self._stream_components(net, sta, loc, band, t_start, t_end)
        if not raw:
            return None, 0

        z  = self._select(raw, ["Z"])
        h1 = self._select(raw, ["N", "1"])
        h2 = self._select(raw, ["E", "2"])

        missing = [name for name, arr in [("Z", z), ("H1", h1), ("H2", h2)]
                   if arr is None]
        if missing:
            seiscomp.logging.warning(
                f"recovar: missing components {missing} for {net}.{sta}.{loc}.{band}"
            )
            return None, 0

        channels = []
        for arr in (z, h1, h2):
            arr = self._fit(arr, n_total)
            if apply_bandpass:
                arr = _bandpass(arr, TARGET_FS)
            channels.append(arr)

        return np.stack(channels, axis=-1).astype(np.float32), pick_idx

    @staticmethod
    def _crop_window(
        data: np.ndarray,
        pick_idx: int,
        offset_s: float,
    ) -> "np.ndarray | None":
        """Extract a 30-second RecovAR window centred at pick_idx + offset_s*FS."""
        n_out  = int((CROP_BEFORE_S + CROP_AFTER_S) * TARGET_FS)   # 3000
        centre = pick_idx + int(offset_s * TARGET_FS)
        start  = centre - int(CROP_BEFORE_S * TARGET_FS)
        end    = start + n_out
        if start < 0 or end > len(data):
            return None
        return data[start:end]

    @staticmethod
    def _score_window(
        raw_data: np.ndarray,
        pick_idx: int,
        offset_s: float,
    ) -> "np.ndarray | None":
        """Produce a (3000, 3) float32 array for scoring using per-window bandpass.

        For a window centred at t_c = t_p + offset_s:
          1. Crop a fresh 40 s window (FETCH_BEFORE_S + FETCH_AFTER_S) from raw_data
          2. Apply 1-20 Hz bandpass to each channel of that isolated window
          3. Crop the inner 30 s (CROP_BEFORE_S + CROP_AFTER_S)
          4. Return (3000, 3) float32

        This prevents signal from other time regions (especially the P-wave) from
        contaminating pre-signal windows through the filter's impulse response.
        Returns None if the window would exceed raw_data bounds.
        """
        n_fetch    = int((FETCH_BEFORE_S + FETCH_AFTER_S) * TARGET_FS)   # 4000
        n_out      = int((CROP_BEFORE_S  + CROP_AFTER_S)  * TARGET_FS)   # 3000
        crop_start = int(FETCH_BEFORE_S * TARGET_FS) - int(CROP_BEFORE_S * TARGET_FS)  # 500

        centre      = pick_idx + int(offset_s * TARGET_FS)
        fetch_start = centre - int(FETCH_BEFORE_S * TARGET_FS)
        fetch_end   = fetch_start + n_fetch

        if fetch_start < 0 or fetch_end > len(raw_data):
            return None

        channels = []
        for ch in range(raw_data.shape[1]):
            window = raw_data[fetch_start:fetch_end, ch]
            window = _bandpass(window, TARGET_FS)
            channels.append(window[crop_start:crop_start + n_out])

        return np.stack(channels, axis=-1).astype(np.float32)

    def _stream_components(self, net, sta, loc, band, t_start, t_end):
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

        buffers: dict[str, list] = {}
        for rec in ri:
            comp_char = rec.channelCode()[-1]
            arr = _record_to_numpy(rec)
            arr = _resample(arr, rec.samplingFrequency(), TARGET_FS)
            buffers.setdefault(comp_char, []).append(arr)

        rs.close()
        return {c: np.concatenate(segs) for c, segs in buffers.items()}

    @staticmethod
    def _select(data: dict, candidates: list) -> "np.ndarray | None":
        for c in candidates:
            if c in data:
                return data[c]
        return None

    @staticmethod
    def _fit(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

    # ------------------------------------------------------------------
    # Attach comments via SeisComP messaging
    # ------------------------------------------------------------------

    def _attach_comment(self, pick, score: float):
        self._send_comment(pick, COMMENT_KEY, f"{COMMENT_KEY}:{score:.4f}")

    def _attach_sweep_comment(self, pick, scores: list):
        start_s = SWEEP_OFFSETS_S[0]
        step_s  = SWEEP_OFFSETS_S[1] - SWEEP_OFFSETS_S[0]
        vals    = ",".join(f"{s:.4f}" for s in scores)
        self._send_comment(pick, SWEEP_COMMENT_KEY,
                           f"{SWEEP_COMMENT_KEY}:{start_s}:{step_s}:{vals}")

    def _send_comment(self, pick, comment_id: str, text: str):
        ci = seiscomp.datamodel.CreationInfo()
        ci.setAgencyID(self.agencyID())
        ci.setAuthor("recovar_pick_filter")
        ci.setCreationTime(seiscomp.core.Time.GMT())

        comment = seiscomp.datamodel.Comment()
        comment.setId(comment_id)   # unique index within the pick
        comment.setText(text)
        comment.setCreationInfo(ci)

        seiscomp.datamodel.Notifier.Enable()
        pick.add(comment)
        msg = seiscomp.datamodel.Notifier.GetMessage()
        seiscomp.datamodel.Notifier.Disable()

        if msg and self.connection():
            self.connection().send("PICK", msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_to_numpy(rec) -> np.ndarray:
    data = rec.data()
    try:
        return np.frombuffer(data.numpy(), dtype=np.float64).copy()
    except (AttributeError, TypeError):
        return np.array([data.get(i) for i in range(data.size())], dtype=np.float64)


def _resample(arr: np.ndarray, src_fs: float, dst_fs: int) -> np.ndarray:
    if abs(src_fs - dst_fs) < 0.01:
        return arr
    from scipy.signal import resample as sp_resample
    n_out = int(round(len(arr) * dst_fs / src_fs))
    return sp_resample(arr, n_out)


def _bandpass(arr: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Fourier-domain Butterworth bandpass (equivalent to filtfilt)."""
    from scipy.signal import butter, freqz
    n     = len(arr)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    b, a  = butter(BP_CORNERS, [BP_LOW_HZ, BP_HIGH_HZ], btype="band", fs=fs)
    _, h  = freqz(b, a, worN=freqs, fs=fs)
    return np.fft.irfft(np.fft.rfft(arr) * np.abs(h) ** 2, n=n)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = RecovARPickFilter(len(sys.argv), sys.argv)
    sys.exit(app())
