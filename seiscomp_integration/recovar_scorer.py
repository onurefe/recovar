"""
recovar_scorer.py — loads the recovar ensemble model once and exposes a
single score() method used by the SeisComP pick filter.
"""

import numpy as np

from recovar.representation_learning_models import RepresentationLearningMultipleAutoencoder
from recovar.classifier_models import ClassifierMultipleAutoencoder

_DUMMY = np.zeros((1, 3000, 3), dtype=np.float32)


class RecovARScorer:
    def __init__(self, model_path: str):
        """
        Load the pre-trained ensemble and build the classifier.

        model_path: path to the .h5 weights file
                    (e.g. models/representation_cross_covariances.h5)
        """
        self._model = RepresentationLearningMultipleAutoencoder(
            name="rep_learning_autoencoder_ensemble",
            input_noise_std=1e-6,
            eps=1e-27,
        )
        self._model.compile()
        # Forward pass required before load_weights to build all sub-layers
        self._model(_DUMMY)
        self._model.load_weights(model_path)

        self._classifier = ClassifierMultipleAutoencoder(self._model)

    def score(self, waveform: np.ndarray) -> float:
        """
        Score a single 3-component waveform.

        waveform: float32 array of shape (3000, 3) — 30 s at 100 Hz,
                  channels ordered Z, N/1, E/2.
        Returns: float in [0, 1] — 1 = seismic signal, 0 = noise.
        """
        x = waveform[np.newaxis].astype(np.float32)  # (1, 3000, 3)
        scores = self._classifier(x)                  # (1,)
        return float(scores[0])
