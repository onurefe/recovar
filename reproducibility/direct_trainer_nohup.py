from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
import os
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)

DATASET='stead'
EPOCH = 10
MODEL = RepresentationLearningMultipleAutoencoder()

trainer = DirectTrainer(dataset=DATASET, 
                            dataset_time_window=STEAD_TIME_WINDOW,
                             model_time_window=WINDOW_SIZE)


history = trainer.train(
    model=model,
    train_dataset_path='preprocessed_data/stead_splits/FULL_DATASET_SUBSAMPLED_1_train.hdf5',
    val_dataset_path='preprocessed_data/stead_splits/FULL_DATASET_SUBSAMPLED_1_val.hdf5',
    test_dataset_path='preprocessed_data/stead_splits/FULL_DATASET_SUBSAMPLED_1_test.hdf5',  # Optional
    epochs=10,
    batch_size=256,
    learning_rate=1e-3,
    use_hdf5_generator=True  
)