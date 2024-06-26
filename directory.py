from os.path import join

HOME_DIR = "."
TRAINED_MODELS_DIR = f"{HOME_DIR}/trained_models"
MONITORING_DIR = f"{HOME_DIR}/monitoring"
RESULTS_DIR = f"{HOME_DIR}/results"
PLOTS_DIR = f"{HOME_DIR}/plots"

STEAD_WAVEFORMS_HDF5_PATH = "/home/onur/stead/waveforms.hdf5"
STEAD_METADATA_CSV_PATH = "/home/onur/stead/metadata.csv"

INSTANCE_NOISE_WAVEFORMS_HDF5_PATH = "/home/onur/instance/noise/waveforms.hdf5"
INSTANCE_EQ_WAVEFORMS_HDF5_PATH = "/home/onur/instance/events/waveforms.hdf5"
INSTANCE_NOISE_METADATA_CSV_PATH = "/home/onur/instance/noise/metadata.csv"
INSTANCE_EQ_METADATA_CSV_PATH = "/home/onur/instance/events/metadata.csv"

PREPROCESSED_DATASET_DIRECTORY = "/home/onur/dataset_preprocessed"


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
    filename = "ep{}.h5".format(epoch)

    return join(checkpoint_dir, filename)


def get_history_csv_path(exp_name, training_model_name, train_dataset, split):
    checkpoint_dir = get_checkpoint_dir(
        exp_name, training_model_name, train_dataset, split
    )

    return join(checkpoint_dir, "history.csv")
