from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder,
                     RepresentationLearningMultipleAutoencoderL4)

from kfold_trainer import KfoldTrainer
from config import KFOLD_SPLITS

# Should be one of the RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

# Should be stead or instance.
DATASETS = ["instance", "stead"]

# Number of epochs
NUM_EPOCHS = 20

# For all splits, train the model over defined datasets.
for eq_ratio in [0.65, 0.75, 0.85, 0.95]:
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
                    resampling_eq_ratio=eq_ratio
                )
                kfold_trainer.train()