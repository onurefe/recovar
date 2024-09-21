import pandas as pd
import numpy as np
from os.path import exists
from sklearn.metrics import (
    roc_curve,
)
from kfold_tester import KFoldTester
from directory import *
from config import SAMPLING_FREQ, PHASE_ENSURING_MARGIN, WINDOW_SIZE

class TracesFilter:
    def __init__(self):
        pass

    def apply(self, df_traces, dataset):
        return df_traces

    def _merge_eq_and_noise_traces(self, df_eq_traces, df_no_traces):
        df_traces = pd.concat([df_eq_traces, df_no_traces], ignore_index=False)
        return df_traces

    def _split_eq_and_noise_traces(self, df_traces):
        df_eq_traces = df_traces[df_traces.label == "eq"]
        df_no_traces = df_traces[df_traces.label == "no"]

        return df_eq_traces, df_no_traces


class SNRFilter(TracesFilter):
    def __init__(self, snr_min=-np.inf, snr_max=np.inf):
        self.snr_min = snr_min
        self.snr_max = snr_max

    def apply(self, df_traces, dataset):
        _df_traces = df_traces.copy()
        df_eq_traces, df_no_traces = self._split_eq_and_noise_traces(_df_traces)

        if dataset == "stead":
            df_eq_traces = self._parse_stead_metadata_enz_snrs(df_eq_traces)

        df_eq_traces["trace_snr"] = self._trace_snr(df_eq_traces)
        df_eq_traces = self._filter_eq_traces(df_eq_traces)

        return self._merge_eq_and_noise_traces(df_eq_traces, df_no_traces)

    def _filter_eq_traces(self, df_eq_traces):
        return df_eq_traces[
            (self.snr_min < df_eq_traces["trace_snr"])
            & (df_eq_traces["trace_snr"] < self.snr_max)
        ]

    @staticmethod
    def _trace_snr(df_eq_traces):
        return 10 * np.log10(
            np.power(10, (df_eq_traces["trace_E_snr_db"].values / 10))
            + np.power(10, (df_eq_traces["trace_N_snr_db"].values / 10))
            + np.power(10, (df_eq_traces["trace_Z_snr_db"].values / 10))
        )

    @staticmethod
    def _parse_stead_metadata_enz_snrs(df_eq_traces):
        _df_eq_traces = df_eq_traces.copy()

        _df_eq_traces["snr_db"] = _df_eq_traces["snr_db"].apply(
            lambda x: SNRFilter._parse_stead_snr(x)
        )
        snr_db_list = np.array(_df_eq_traces["snr_db"].values.tolist())

        _df_eq_traces["trace_E_snr_db"] = snr_db_list[:, 0]
        _df_eq_traces["trace_N_snr_db"] = snr_db_list[:, 1]
        _df_eq_traces["trace_Z_snr_db"] = snr_db_list[:, 2]

        return _df_eq_traces

    @staticmethod
    def _parse_stead_snr(printedlist):
        printedlist = printedlist.replace("[", "")
        printedlist = printedlist.replace("]", "")
        printedlist = printedlist.split(" ")
        slicer = np.array([x.replace(" ", "") != "" for x in printedlist])

        return [float(x) for x in np.array(printedlist)[slicer]]


class CropOffsetFilter(TracesFilter):
    def __init__(
        self,
        min_before_p_arrival_in_seconds=PHASE_ENSURING_MARGIN,
        min_after_p_arrival_in_seconds=PHASE_ENSURING_MARGIN,
        window_size_in_seconds=WINDOW_SIZE,
        sampling_frequency=SAMPLING_FREQ,
    ):
        self.min_before_p_arrival_in_samples = int(
            sampling_frequency * min_before_p_arrival_in_seconds
        )
        self.min_after_p_arrival_in_samples = int(
            sampling_frequency * min_after_p_arrival_in_seconds
        )
        self.window_size_in_samples = int(sampling_frequency * window_size_in_seconds)

    def apply(self, df_traces):
        _df_traces = df_traces.copy()
        df_eq_traces, df_no_traces = self._split_eq_and_noise_traces(_df_traces)

        df_eq_traces = self._filter_eq_traces(df_eq_traces)

        return self._merge_eq_and_noise_traces(df_eq_traces, df_no_traces)

    def _filter_eq_traces(self, df_eq_traces):
        min_crop_offset = (
            df_eq_traces["p_arrival_sample"]
            + self.min_after_p_arrival_in_samples
            - self.window_size_in_samples
        )
        max_crop_offset = (
            df_eq_traces["p_arrival_sample"] - self.min_before_p_arrival_in_samples
        )

        return df_eq_traces[
            (
                (min_crop_offset < df_eq_traces["crop_offset"])
                & ((df_eq_traces["crop_offset"] < max_crop_offset))
            )
        ]


class Evaluator:
    def __init__(
        self,
        exp_name,
        representation_learning_model_class,
        classifier_model_class,
        train_dataset,
        test_dataset,
        filters,
        epochs,
        split,
        method_params
    ):
        self.exp_name = exp_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.filters = filters
        self.epochs = epochs
        self.split = split
        self.method_params = method_params
        self.representation_learning_model_name = representation_learning_model_class().name
        self.classifier_model_name = classifier_model_class().name

        self._add_tester(
            representation_learning_model_class=representation_learning_model_class,
            classifier_model_class=classifier_model_class,
        )

    def get_roc_vectors(
        self,
    ):
        if not self._are_results_present():
            self.tester.test()

        df_meta = self._read_meta_file()
        df_meta = self._apply_filters(df_meta)
        indexer = np.array(df_meta.index)

        roc_vectors = []
        for epoch in self.epochs:
            df_score = self._read_score_file(epoch)
            df_score = df_score.iloc[indexer]
            tpr, fpr, thresholds = self._get_roc_vector(
                df_score, df_meta
            )
            roc_vectors.append({"tpr": tpr, "fpr": fpr, "thresholds": thresholds})

        return roc_vectors

    def _get_roc_vector(self, df_score, df_meta):
        labels = df_meta["label"] == "eq"
        fpr, tpr, thresholds = roc_curve(labels, df_score["eq_probabilities"])

        return tpr, fpr, thresholds

    def _apply_filters(self, metadata):
        _metadata = metadata.copy()
        _metadata.reset_index(inplace=True)

        for filter in self.filters:
            _metadata = filter.apply(_metadata)

        _metadata = _metadata.sample(frac=1)
        return _metadata

    def _read_meta_file(self):
        return pd.read_csv(self._get_meta_file_path())

    def _read_score_file(self, epoch):
        return pd.read_csv(self._get_score_file_path(epoch))

    def _add_tester(self, representation_learning_model_class, classifier_model_class):
        self.tester = KFoldTester(
            self.exp_name,
            representation_learning_model_class=representation_learning_model_class,
            classifier_model_class=classifier_model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            split=self.split,
            epochs=self.epochs,
            method_params=self.method_params,
        )

    def _are_results_present(self):
        return self._are_score_files_present() and self._is_meta_file_present()

    def _are_score_files_present(self):
        score_file_exists = []

        for epoch in self.epochs:
            score_file_exists.append(exists(self._get_score_file_path(epoch)))

        return np.array(score_file_exists).all()

    def _is_meta_file_present(self):
        return exists(self._get_meta_file_path())

    def _get_score_file_path(self, epoch):
        score_file_path = get_exp_results_score_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
            epoch,
        )

        return score_file_path

    def _get_meta_file_path(self):
        meta_file_path = get_exp_results_meta_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )

        return meta_file_path
