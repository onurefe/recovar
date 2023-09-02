# Outputs are stored in the following directories.
HOME_DIR = "."
TRAINED_MODELS_DIR = f"{HOME_DIR}/trained_models"
MONITORING_DIR = f"{HOME_DIR}/monitoring"
RESULTS_DIR = f"{HOME_DIR}/results"

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