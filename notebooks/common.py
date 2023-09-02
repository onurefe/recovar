import numpy as np
from scipy import signal


def method_aliases(method_name):
    mapper = {
        "AutocovarianceMonitor30s": "Autocovariance",
        "EnsembleAugmentationCrossCovarianceMonitor30s": "Augmentation Cross-covariances",
        "Ensemble5CrossCovarianceMonitor30s": "Representation Cross-covariances",
        "StaLtaMonitor": "STA-LTA",
    }
    return mapper[method_name]


def aliases_to_method_names(method_alias):
    mapper = {
        "Autocovariance": "AutocovarianceMonitor30s",
        "Augmentation Ensemble": "EnsembleAugmentationCrossCovarianceMonitor30s",
        "Augmentation Cross-covariances": "EnsembleAugmentationCrossCovarianceMonitor30s",
        "Representation Ensemble": "Ensemble5CrossCovarianceMonitor30s",
        "Representation Cross-covariances": "Ensemble5CrossCovarianceMonitor30s",
        "STA-LTA": "StaLtaMonitor",
    }
    return mapper[method_alias]


def method_fname(method_name):
    mapper = {
        "AutocovarianceMonitor30s": "autocovariance_30s",
        "EnsembleAugmentationCrossCovarianceMonitor30s": "augmentation_ensemble_30s",
        "Ensemble5CrossCovarianceMonitor30s": "representation_ensemble_30s",
        "EnsembleAndAugmentationCrossCovarianceMonitor30s": "representation_augmentation_ensemble_30s",
        "StaLtaMonitor": "sta_lta",
        "LatentStaLtaMonitor": "latent_sta_lta",
    }
    return mapper[method_name]


def get_traces_path(results_dir, split_idx, train_dataset, test_dataset, model_name):
    return "{}/training_{}/testing_{}/{}/split{}/traces.csv".format(
        results_dir, train_dataset, test_dataset, model_name, split_idx
    )


def get_monitor_path(
    results_dir,
    split_idx,
    train_dataset,
    test_dataset,
    model_name,
    epoch,
    monitor_params,
):
    return "{}/training_{}/testing_{}/{}/split{}/epoch{}_monitoredparams{}.hdf5".format(
        results_dir,
        train_dataset,
        test_dataset,
        model_name,
        split_idx,
        epoch,
        monitor_params,
    )


def trace_snr(df):
    return 10 * np.log10(
        np.power(10, (df["trace_E_snr_db"].values / 10))
        + np.power(10, (df["trace_N_snr_db"].values / 10))
        + np.power(10, (df["trace_Z_snr_db"].values / 10))
    )


def cropoffset_filter(df, before_margin=300, after_margin=300, window_size=3000):
    # Filter for crop offsets.
    min_crop_offset = df["p_arrival_sample"] + after_margin - window_size
    max_crop_offset = df["p_arrival_sample"] - before_margin

    return df[
        (
            (min_crop_offset < df["crop_offset"])
            & ((df["crop_offset"] < max_crop_offset))
        )
    ]


def snr_filter(df, snr_min=-np.inf, snr_max=np.inf):
    _df = df.copy()
    _df["trace_snr"] = trace_snr(_df)
    return _df[(snr_min < _df["trace_snr"]) & (_df["trace_snr"] < snr_max)]


def parse_stead_snr(printedlist):
    printedlist = printedlist.replace("[", "")
    printedlist = printedlist.replace("]", "")
    printedlist = printedlist.split(" ")
    slicer = np.array([x.replace(" ", "") != "" for x in printedlist])

    return [float(x) for x in np.array(printedlist)[slicer]]


def get_maxf1_threshold(thresholds, pre, rec):
    f1_score = 2 * pre * rec / (1e-37 + pre + rec)
    max_f1_score_idx = np.argmax(f1_score)
    return thresholds[max_f1_score_idx]


def get_metrics(pre, rec, thresholds, best_f1_threshold):
    idx = np.argmin(np.abs(thresholds - best_f1_threshold))
    f1_score = 2 * pre * rec / (1e-37 + pre + rec)
    return {"pre": pre[idx], "rec": rec[idx], "f1_score": f1_score[idx]}


def gaussian(x, mu, sig, axis=1):
    g = np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
    return g / np.sum(g, axis=axis, keepdims=True)


def detector(detection_vector, detection_vector_name):
    if detection_vector_name == "xcov" or detection_vector_name == "ycov":
        window_size = 5.0

        ts = np.shape(detection_vector)[1]
        t = np.expand_dims(np.linspace(-15, 15, ts), axis=0)

        g = gaussian(t, np.zeros_like(t), window_size * np.ones_like(t))
        result = np.sum(np.maximum(detection_vector, 0) * g, axis=1)
    elif detection_vector_name == "fcov":
        window_size = 2.5
        ts = np.shape(detection_vector)[1]
        t = np.expand_dims(np.linspace(-15, 15, ts), axis=0)
        g = gaussian(t, np.zeros_like(t), window_size * np.ones_like(t))
        result = np.maximum(np.sum(detection_vector * g, axis=1), 0)
    elif detection_vector_name == "fccov":
        result = detection_vector
    elif detection_vector_name == "score":
        result = detection_vector
    else:
        raise ValueError("Unknown evaluation metric: {}".format(detection_vector_name))

    return result
