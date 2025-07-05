from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)


DATASET = 'stead'
MODEL = RepresentationLearningMultipleAutoencoder()
EPOCH = 10

trainer = DirectTrainer(dataset=DATASET, 
                            dataset_time_window=STEAD_TIME_WINDOW,
                             model_time_window=WINDOW_SIZE)


trainer.create_subsampled_datasets(
    dataset='stead',
    output_dir='preprocessed_data/stead_splits',
    noise_percentages=[], 
    subsampling_factor=1, 
    maintain_constant_size=False,
    save_train_val_test_splits=True, 
    val_ratio=0.2, 
    test_ratio=0.2,
    random_state_mode='pseudorandom',
    base_random_state=42
)