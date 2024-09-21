from os.path import join
import json
import os

# Load the JSON data
with open("settings.json", 'r') as file:
    settings = json.load(file)
    
STEAD_WAVEFORMS_HDF5_PATH = settings["DATASET_DIRECTORIES"]["STEAD_WAVEFORMS_HDF5_PATH"]
STEAD_METADATA_CSV_PATH = settings["DATASET_DIRECTORIES"]["STEAD_METADATA_CSV_PATH"]
INSTANCE_NOISE_WAVEFORMS_HDF5_PATH = settings["DATASET_DIRECTORIES"]["INSTANCE_NOISE_WAVEFORMS_HDF5_PATH"]
INSTANCE_EQ_WAVEFORMS_HDF5_PATH = settings["DATASET_DIRECTORIES"]["INSTANCE_EQ_WAVEFORMS_HDF5_PATH"]
INSTANCE_NOISE_METADATA_CSV_PATH = settings["DATASET_DIRECTORIES"]["INSTANCE_NOISE_METADATA_CSV_PATH"]
INSTANCE_EQ_METADATA_CSV_PATH = settings["DATASET_DIRECTORIES"]["INSTANCE_EQ_METADATA_CSV_PATH"]

PREPROCESSED_DATASET_DIRECTORY = settings["PREPROCESSED_DATASET_DIRECTORY"]
RESULTS_DIR = settings["RESULTS_DIR"]
TRAINED_MODELS_DIR = settings["TRAINED_MODELS_DIR"]

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
    output_dir = get_exp_results_dir(
        exp_name,
        representation_learning_model_name,
        classification_model_name,
        train_dataset,
        test_dataset,
        split
    )

    filename = "scores{}.csv".format(epoch)

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
