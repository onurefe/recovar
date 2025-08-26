from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder)

from kfold_trainer import KfoldTrainer
from config import KFOLD_SPLITS

# Should be one of the RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

# Should be stead or instance.
DATASETS = ["stead", "instance"]

# Number of epochs
NUM_EPOCHS = 100

# For all splits, train the model over defined datasets.
for eq_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for train_dataset in DATASETS:
        for model_class in MODEL_CLASSES:
            for split in range(KFOLD_SPLITS):
                exp_name = f"exp_resample_eq_ratio{eq_ratio}"
                kfold_trainer = KfoldTrainer(
                    exp_name, 
                    model_class, 
                    train_dataset, 
                    split, 
                    epochs=NUM_EPOCHS, 
                    apply_resampling=True, 
                    resample_while_keeping_total_waveforms_fixed=True,
                    resampling_eq_ratio=eq_ratio
                )
                kfold_trainer.train()