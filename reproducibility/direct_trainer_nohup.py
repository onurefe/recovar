from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
import os
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)


subsampled_h5s_dir = "/home/ege/recovar_data_preprocessed/stead_pseudorandom_state_ne_subsampled"
EPOCH = 1
MODEL = RepresentationLearningSingleAutoencoder()

for (root, folder, file) in os.walk(subsampled_h5s_dir):

    trainer = DirectTrainer(dataset='stead', 
    dataset_time_window=STEAD_TIME_WINDOW, model_time_window=WINDOW_SIZE)

    model = MODEL

    history = trainer.train(
        model=model,
        train_dataset_path="/home/ege/recovar_data_preprocessed/SUBSAMPLED_100_NOISE_10.hdf5",
        val_dataset_path="/home/ege/recovar_data_preprocessed/SUBSAMPLED_100_NOISE_10.hdf5",
        epochs=EPOCH
    )