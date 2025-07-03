from direct_trainer import DirectTrainer
from recovar import RepresentationLearningSingleAutoencoder, RepresentationLearningDenoisingSingleAutoencoder, RepresentationLearningMultipleAutoencoder

from directory import (
    STEAD_WAVEFORMS_HDF5_PATH,
    STEAD_METADATA_CSV_PATH,
    INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
    INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
    INSTANCE_EQ_METADATA_CSV_PATH,
    INSTANCE_NOISE_METADATA_CSV_PATH,
    PREPROCESSED_DATASET_DIRECTORY,
)

trainer = DirectTrainer(dataset='stead',
    dataset_time_window=60.0,  # STEAD has 60s windows
    model_time_window=30.0
)

trainer.create_subsampled_datasets(
    dataset='stead',
    output_dir=PREPROCESSED_DATASET_DIRECTORY,
    noise_percentages=[],
    subsampling_factor=1.0, 
    maintain_constant_size=False
)
