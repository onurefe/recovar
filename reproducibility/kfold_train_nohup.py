from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
from kfold_trainer import KfoldTrainer
from config import KFOLD_SPLITS

# Experiment name.
EXP_NAME = "continuous_20"

# Should be one of the RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
MODEL_CLASSES = [RepresentationLearningMultipleAutoencoder]#, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder]

# Should be stead or instance.
DATASETS = ["continuous"]

# Number of epochs
NUM_EPOCHS = 20

# For all splits, train the model over defined datasets.
for train_dataset in DATASETS:
    for model_class in MODEL_CLASSES:
        for split in range(KFOLD_SPLITS):
            kfold_trainer = KfoldTrainer(
                EXP_NAME, model_class, train_dataset, split, epochs=NUM_EPOCHS
            )
            kfold_trainer.train()