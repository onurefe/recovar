from recovar import (RepresentationLearningSingleAutoencoder, 
                     RepresentationLearningDenoisingSingleAutoencoder, 
                     RepresentationLearningMultipleAutoencoder)

from kfold_trainer import KfoldTrainer
from config import KFOLD_SPLITS

# Should be one of the RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]

# Should be stead, instance, or any custom dataset defined in settings.json
DATASETS = ["erik_fixed"]

# Number of epochs
NUM_EPOCHS = 20
# For all splits, train the model over defined datasets.
for eq_ratio in [0.01, 0.02, 0.03, 0.04, 0.05]:
    for train_dataset in DATASETS:
        for model_class in MODEL_CLASSES:
            for split in range(1):
                exp_name = f"exp_{train_dataset}_resample_eq_ratio_{eq_ratio}"
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