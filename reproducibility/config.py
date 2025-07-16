import json
import sys

# Load the JSON data
with open("settings.json", 'r') as file:
    settings = json.load(file)

""" After splitting the dataset into chunks, we split each chunk into training, validation and test sets.
This ratio is the ratio of training set size to the validation set size. The data is first split
into test and train+validation sets. Then the train+validation set is split into train and validation
sets. This is required since Kfold split is used. """        
SUBSAMPLING_FACTOR = settings["CONFIG"]["SUBSAMPLING_FACTOR"]
TRAIN_VALIDATION_SPLIT = settings["CONFIG"]["TRAIN_VALIDATION_SPLIT"]
KFOLD_SPLITS = settings["CONFIG"]["KFOLD_SPLITS"]
DATASET_CHUNKS = settings["CONFIG"]["DATASET_CHUNKS"]
PHASE_PICK_ENSURED_CROP_RATIO = settings["CONFIG"]["PHASE_PICK_ENSURED_CROP_RATIO"]
PHASE_ENSURING_MARGIN = settings["CONFIG"]["PHASE_ENSURING_MARGIN"]
TEST_RATIO = settings["CONFIG"]["SINGLE_SPLIT_TEST_RATIO"]
APPLY_RESAMPLING = settings["CONFIG"]["APPLY_RESAMPLING"]
RESAMPLE_EQ_RATIO = settings["CONFIG"]["RESAMPLE_EQ_RATIO"]

# Batch size. This is common for all models. Should be kept as is.
WINDOW_SIZE = 30.

# The time window of the data. Different datasets have different time windows.
INSTANCE_TIME_WINDOW = 120.0
STEAD_TIME_WINDOW = 60.0

# The frequency range of the bandpass filter. The bandpass filter is applied to the data before
# training, testing and validation. The filter is applied to the data in the preprocessing step.
FREQMIN = 1.0
FREQMAX = 20.0

# The number of channels in the data. The data is multichannel.
N_CHANNELS = 3

# Batch size. This is common for all models. Should be kept as is.
BATCH_SIZE = 256

# Sampling frequency. This is a common parameter for all models. Should be kept as is.
SAMPLING_FREQ = 100