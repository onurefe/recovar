from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
import os
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)

DATASET='stead'
EPOCH = 10
MODEL = RepresentationLearningMultipleAutoencoder()

trainer = DirectTrainer(dataset=DATASET, 
dataset_time_window=STEAD_TIME_WINDOW, model_time_window=WINDOW_SIZE)


history = trainer.train(
    model=MODEL,
    dataset_path="/home/ege/recovar_data_preprocessed/FULL_DATASET_SUBSAMPLED_100.hdf5",
    epochs=EPOCH
)