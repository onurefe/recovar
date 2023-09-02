from training_models import (
    AutocovarianceDetector30s,
    AutocovarianceDetectorDenoising30s,
    Ensemble5CrossCovarianceDetector30s,
    AutocovarianceDetectorAttentionDenoising30s,
)
from kfold_trainer import Trainer

kfold_trainer = Trainer(Ensemble5CrossCovarianceDetector30s)
kfold_trainer.train(
    dataset="instance", epochs=20, splits=[0, 1, 2, 3, 4], starting_epoch=0
)

kfold_trainer = Trainer(Ensemble5CrossCovarianceDetector30s)
kfold_trainer.train(
    dataset="stead", epochs=20, splits=[0, 1, 2, 3, 4], starting_epoch=0
)
