from training_models import (
    AutocovarianceDetector30s,
    AutocovarianceDetectorDenoising30s,
    StaLtaDetector,
    TrainableStaLtaDetector,
    Ensemble5CrossCovarianceDetector30s,
)
from monitor_models import (
    AutocovarianceMonitor30s,
    StaLtaMonitor,
    EnsembleAugmentationCrossCovarianceMonitor30s,
    Ensemble5CrossCovarianceMonitor30s,
)
from kfold_environment import KFoldEnvironment
from kfold_tester import KFoldTester

DETECTOR_CTOR = AutocovarianceDetectorDenoising30s
MONITOR_CTOR = EnsembleAugmentationCrossCovarianceMonitor30s
MONITOR_PARAMS = ["fcov"]
LOAD_MODEL = True
EXP_NAMES = [
    "augmentation_ensemble_denoising",
]
METHOD_PARAMS = [
    {
        "augmentations": 5,
        "noise_augmentation_params": {"weight": 1e-3, "enabled": False},
        "dispersion_augmentation_params": {
            "distance_min": 0.5e5,
            "distance_max": 0.6e5,
            "phase_velocity_min": 8e3,
            "phase_velocity_max": 10e3,
            "phase_velocity_std": 1e3,
            "knots": 4,
            "enabled": False,
        },
        "timewarping_params": {"std": 0.15, "knots": 4, "enabled": True},
    }
]

SPLITS = [0, 1, 2, 3, 4]

dataset_pairs = [
    ["instance", "instance"],
    ["instance", "stead"],
    ["stead", "instance"],
    ["stead", "stead"],
]
epochs = range(-1, 20)

for i in range(len(EXP_NAMES)):
    for dataset_pair in dataset_pairs:
        # Form train environment.
        train_kenv = KFoldEnvironment(
            dataset=dataset_pair[0],
            subsampling_factor=1.0,
        )

        # Form test environment.
        test_kenv = KFoldEnvironment(
            dataset=dataset_pair[1],
            subsampling_factor=1.0,
        )
        for epoch in epochs:
            # Create a tester object. Constructors from detector and monitor models should be specified.
            tester = KFoldTester(
                train_environment=train_kenv,
                test_environment=test_kenv,
                detector_ctor=DETECTOR_CTOR,
                monitor_ctor=MONITOR_CTOR,
            )

            # Test the model on folds 0. Monitoring results will be stored on
            # {HOME_DIR}/results{exp_name}/training_{train_dataset}/test_{test_dataset}/{monitor_model_name}/split_{split_number}/epoch{epoch}_monitoredparams{monitor_params}.hdf5
            # Besides at the same directory traces.csv file is generated. This file contains the traces of the monitored parameters in the same order with the hdf5 file.
            METHOD_PARAMS[i]["epoch"] = epoch
            METHOD_PARAMS[i]["load_model"] = LOAD_MODEL

            tester.test(
                exp_name=EXP_NAMES[i],
                splits=SPLITS,
                monitor_params=MONITOR_PARAMS,
                method_params=METHOD_PARAMS[i],
            )
