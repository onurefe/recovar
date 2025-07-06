from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
import os
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)

DATASET='instance'
EPOCH = 10
MODEL = RepresentationLearningMultipleAutoencoder()

trainer = DirectTrainer(dataset=DATASET, 
                            dataset_time_window=INSTANCE_TIME_WINDOW,
                             model_time_window=WINDOW_SIZE)


history = trainer.train(
    model=MODEL,
    train_dataset_path='/home/ege/recovar/reproducibility/preprocessed_data/new_instance/FULL_INSTANCE_SUBSAMPLED_100_train.hdf5',
    val_dataset_path='/home/ege/recovar/reproducibility/preprocessed_data/new_instance/FULL_INSTANCE_SUBSAMPLED_100_val.hdf5',
    test_dataset_path='/home/ege/recovar/reproducibility/preprocessed_data/new_instance/FULL_INSTANCE_SUBSAMPLED_100_test.hdf5',
    epochs=EPOCH,
    batch_size=256,
    learning_rate=1e-3,
    use_hdf5_generator=True  
)