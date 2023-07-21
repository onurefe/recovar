# Batch size. This is common for all models. Should be kept as is.
BATCH_SIZE = 256

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

# Outputs are stored in the following directories.
HOME_DIR = "."
TRAINED_MODELS_DIR = f"{HOME_DIR}/trained_models"
MONITORING_DIR = f"{HOME_DIR}/monitoring"

# The path of the hdf5 file for the STEAD dataset that contains the waveforms of the events.
STEAD_WAVEFORMS_HDF5_PATH = "/home/onur/stead/waveforms.hdf5"
STEAD_METADATA_CSV_PATH = "/home/onur/stead/metadata.csv"

# The path of the hdf5 file for the instance dataset that contains the waveforms of the events.
INSTANCE_NOISE_WAVEFORMS_HDF5_PATH = "/home/onur/instance/noise/waveforms.hdf5"
INSTANCE_EQ_WAVEFORMS_HDF5_PATH = "/home/onur/instance/events/waveforms.hdf5"
INSTANCE_NOISE_METADATA_CSV_PATH = "/home/onur/instance/noise/metadata.csv"
INSTANCE_EQ_METADATA_CSV_PATH = "/home/onur/instance/events/metadata.csv"

# Preprocessed data is saved in this directory. The preprocessed data is used for training, testing and validation.
PREPROCESSED_DATASET_DIRECTORY = "/home/onur/dataset_preprocessed"
