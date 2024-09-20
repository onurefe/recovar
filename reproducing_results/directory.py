from os.path import join

HOME_DIR = "."
STEAD_DIR = "./data/stead"
INSTANCE_DIR = "./data/instance"
TRAINED_MODELS_DIR = f"{HOME_DIR}/trained_models"
RESULTS_DIR = f"{HOME_DIR}/results"
PLOTS_DIR = f"{HOME_DIR}/plots"

STEAD_WAVEFORMS_HDF5_PATH = f"{STEAD_DIR}/waveforms.hdf5"
STEAD_METADATA_CSV_PATH = f"{STEAD_DIR}/metadata.csv"

INSTANCE_NOISE_WAVEFORMS_HDF5_PATH = f"{INSTANCE_DIR}/noise/waveforms.hdf5"
INSTANCE_EQ_WAVEFORMS_HDF5_PATH = f"{INSTANCE_DIR}/events/waveforms.hdf5"
INSTANCE_NOISE_METADATA_CSV_PATH = f"{INSTANCE_DIR}/noise/metadata.csv"
INSTANCE_EQ_METADATA_CSV_PATH = f"{INSTANCE_DIR}/events/metadata.csv"

def get_exp_results_dir(
    exp_name,
    representation_learning_model_name,
    classification_model_name,
    train_dataset,
    test_dataset,
    split,
):
    return join(
        RESULTS_DIR,
        exp_name,
        representation_learning_model_name,
        classification_model_name,
        "training_" + train_dataset,
        "testing_" + test_dataset,
        "split" + str(split),
    )


def get_exp_results_meta_file_path(
    exp_name,
    representation_learning_model_name,
    classification_model_name,
    train_dataset,
    test_dataset,
    split,
):
    output_dir = get_exp_results_dir(
        exp_name,
        representation_learning_model_name,
        classification_model_name,
        train_dataset,
        test_dataset,
        split,
    )

    return join(output_dir, "meta.csv")


def get_exp_results_score_file_path(
    exp_name,
    representation_learning_model_name,
    classification_model_name,
    train_dataset,
    test_dataset,
    split,
    epoch,
):
    output_dir = get_exp_results_score_file_path(
        exp_name,
        representation_learning_model_name,
        classification_model_name,
        train_dataset,
        test_dataset,
        split,
    )

    filename = "epoch{}.hdf5".format(epoch)

    return join(output_dir, filename)


def get_checkpoint_dir(exp_name, representation_learning_model_name, train_dataset, split):
    return join(
        TRAINED_MODELS_DIR,
        exp_name,
        representation_learning_model_name,
        train_dataset,
        "split" + str(split),
    )


def get_checkpoint_path(exp_name, representation_learning_model_name, train_dataset, split, epoch):
    checkpoint_dir = get_checkpoint_dir(
        exp_name, representation_learning_model_name, train_dataset, split
    )
    filename = "ep{}.h5".format(epoch)

    return join(checkpoint_dir, filename)


def get_history_csv_path(exp_name, representation_learning_model_name, train_dataset, split):
    checkpoint_dir = get_checkpoint_dir(
        exp_name, representation_learning_model_name, train_dataset, split
    )

    return join(checkpoint_dir, "history.csv")
