from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder
from config import (STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW, WINDOW_SIZE)


DATASET = 'stead'
trainer = DirectTrainer(dataset=DATASET, 
                            dataset_time_window=STEAD_TIME_WINDOW,
                             model_time_window=WINDOW_SIZE)


# Reduced chunk_size for very large datasets
# 20000 samples ≈ 7.2 GB at processing time, 10000 ≈ 3.6 GB
trainer.create_subsampled_datasets(
    dataset='stead',
    output_dir='preprocessed_data/new_stead',
    noise_percentages=[], 
    subsampling_factor=1, 
    maintain_constant_size=False,
    save_train_val_test_splits=True, 
    val_ratio=0.2, 
    test_ratio=0.2,
    random_state_mode='pseudorandom',
    base_random_state=42,
    chunk_size=50000 
)