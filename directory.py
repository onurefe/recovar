from os.path import join
from os.path import exists
from os import makedirs

# We need to retry with HOME_DIR='.', last time, there were some issues with it. So it is better to localize it for now.
HOME_DIR = "/home/ege/Documents/EARTH-ML/LatentCovarianceBasedSeismicEventDetection/main/"
ALL_DIR = 'latcov_all_data'

TRAINED_MODELS_DIR = join(HOME_DIR, ALL_DIR, 'trained_models')
MONITORING_DIR = join(HOME_DIR, ALL_DIR, 'monitoring')
RESULTS_DIR = join(HOME_DIR, ALL_DIR, 'results')

STEAD_DIR = join(HOME_DIR, ALL_DIR, 'stead')
STEAD_WAVEFORMS_HDF5_PATH = join(STEAD_DIR, 'waveforms.hdf5')
STEAD_METADATA_CSV_PATH = join(STEAD_DIR, 'metadata.csv')


INSTANCE_DIR = join(HOME_DIR, ALL_DIR, 'instance')
NOISE_DIR = join(INSTANCE_DIR, 'noise')
EVENTS_DIR = join(INSTANCE_DIR, 'events')

INSTANCE_NOISE_WAVEFORMS_HDF5_PATH = join(NOISE_DIR, 'waveforms.hdf5')
INSTANCE_NOISE_METADATA_CSV_PATH = join(NOISE_DIR, 'metadata.csv')

INSTANCE_EQ_WAVEFORMS_HDF5_PATH = join(EVENTS_DIR, 'waveforms.hdf5')
INSTANCE_EQ_METADATA_CSV_PATH = join(EVENTS_DIR, 'metadata.csv')


PREPROCESSED_DATASET_DIRECTORY = join(HOME_DIR, ALL_DIR, 'dataset_preprocessed')

def initiate_dirs(dirs=[ALL_DIR,TRAINED_MODELS_DIR, MONITORING_DIR, RESULTS_DIR, STEAD_DIR, INSTANCE_DIR,NOISE_DIR,EVENTS_DIR,PREPROCESSED_DATASET_DIRECTORY]):
    for dir in dirs: 
        if not exists(dir):
            makedirs(dir)


def get_monitoring_output_dir(
    exp_name,
    training_model_name,
    monitoring_model_name,
    train_dataset,
    test_dataset,
    split,
):
    return join(
        MONITORING_DIR,
        exp_name,
        monitoring_model_name,
        training_model_name,
        "training_" + train_dataset,
        "testing_" + test_dataset,
        "split" + str(split),
    )


def get_monitoring_meta_file_path(
    exp_name,
    training_model_name,
    monitoring_model_name,
    train_dataset,
    test_dataset,
    split,
):
    monitoring_output_dir = get_monitoring_output_dir(
        exp_name,
        training_model_name,
        monitoring_model_name,
        train_dataset,
        test_dataset,
        split,
    )

    return join(monitoring_output_dir, "meta.csv")


def get_monitoring_data_file_path(
    exp_name,
    training_model_name,
    monitoring_model_name,
    train_dataset,
    test_dataset,
    split,
    epoch,
    monitored_params,
):
    monitoring_output_dir = get_monitoring_output_dir(
        exp_name,
        training_model_name,
        monitoring_model_name,
        train_dataset,
        test_dataset,
        split,
    )

    filename = "epoch{}_monitoredparams{}.hdf5".format(epoch, monitored_params)

    return join(monitoring_output_dir, filename)


def get_checkpoint_dir(exp_name, training_model_name, train_dataset, split):
    return join(
        TRAINED_MODELS_DIR,
        exp_name,
        training_model_name,
        train_dataset,
        "split" + str(split),
    )


def get_checkpoint_path(exp_name, training_model_name, train_dataset, split, epoch):
    checkpoint_dir = get_checkpoint_dir(
        exp_name, training_model_name, train_dataset, split
    )
    filename = "ep{}.weights.h5".format(epoch)

    return join(checkpoint_dir, filename)


def get_history_csv_path(exp_name, training_model_name, train_dataset, split):
    checkpoint_dir = get_checkpoint_dir(
        exp_name, training_model_name, train_dataset, split
    )

    return join(checkpoint_dir, "history.csv")
