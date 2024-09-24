from os.path import join
import json
import os

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
        
# Batch size. This is common for all models. Should be kept as is.
BATCH_SIZE = 256
WINDOW_SIZE = 30.

# Sampling frequency. This is a common parameter for all models. Should be kept as is.
SAMPLING_FREQ = 100

# The time window of the data. Different datasets have different time windows.
INSTANCE_TIME_WINDOW = 120.0
STEAD_TIME_WINDOW = 60.0

# Change this factor if you want to use less data for training, testing and validation.
SUBSAMPLING_FACTOR = 1.0

# After splitting the dataset into chunks, we split each chunk into training, validation and test sets.
# This ratio is the ratio of training set size to the validation set size. The data is first split
# into test and train+validation sets. Then the train+validation set is split into train and validation
# sets. This is required since Kfold split is used.
TRAIN_VALIDATION_SPLIT = 3.0 / 4
KFOLD_SPLITS = 5
DATASET_CHUNKS = 20

# Ratio of the samples which at least one pick(P or S) is ensured to be included in the window. This
# ratio is used just for training.
PHASE_PICK_ENSURED_CROP_RATIO = 2.0 / 3

# Margin(in seconds) which is used to ensure that the picks are included in the window.
PHASE_ENSURING_MARGIN = 3.0

# The frequency range of the bandpass filter. The bandpass filter is applied to the data before
# training, testing and validation. The filter is applied to the data in the preprocessing step.
FREQMIN = 1.0
FREQMAX = 20.0

# The number of channels in the data. The data is multichannel.
N_CHANNELS = 3