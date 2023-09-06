import pandas as pd
import numpy as np
import h5py as h5
from os.path import exists
from kfold_tester import KFoldTester
from directory import *
from sklearn.metrics import (
    roc_curve,
)


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
        min_before_p_arrival_in_seconds,
        min_after_p_arrival_in_seconds,
        window_size_in_seconds,
        sampling_frequency,
    ):
        self.min_before_p_arrival_in_samples = int(
            sampling_frequency * min_before_p_arrival_in_seconds
        )
        self.min_after_p_arrival_in_samples = int(
            sampling_frequency * min_after_p_arrival_in_seconds
        )
        self.window_size_in_samples = int(sampling_frequency * window_size_in_seconds)

    def apply(self, df_traces, dataset):
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
        training_model_ctor,
        monitoring_model_ctor,
        train_dataset,
        test_dataset,
        filters,
        epochs,
        split,
        method_params,
        metric,
    ):
        self.exp_name = exp_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.filters = filters
        self.epochs = epochs
        self.split = split
        self.method_params = method_params
        self.metric = metric
        self.training_model_name = training_model_ctor().name
        self.monitoring_model_name = monitoring_model_ctor().name

        self._add_tester(
            training_model_ctor=training_model_ctor,
            monitoring_model_ctor=monitoring_model_ctor,
        )

    def get_roc_vectors(
        self,
    ):
        if not self._are_monitoring_files_present():
            self.tester.test()

        monitoring_meta = self._read_monitoring_meta()
        monitoring_meta = self._apply_filters(monitoring_meta)
        indexer = np.array(monitoring_meta.index)

        roc_vectors = []
        for epoch in self.epochs:
            monitoring_data = self._read_monitoring_data(epoch)
            monitoring_data = self._slice_monitoring_tensors(monitoring_data, indexer)
            tpr, fpr, thresholds = self._get_roc_vector(
                monitoring_data, monitoring_meta
            )
            roc_vectors.append({"tpr": tpr, "fpr": fpr, "thresholds": thresholds})

        return roc_vectors

    def _get_roc_vector(self, monitoring_data, monitoring_meta):
        metric = self.metric(monitoring_data)
        labels = monitoring_meta["label"] == "eq"
        fpr, tpr, thresholds = roc_curve(labels, metric)

        return tpr, fpr, thresholds

    def _slice_monitoring_tensors(self, monitoring_data, indexer):
        for key in self.metric.monitored_params:
            monitoring_data[key] = monitoring_data[key][indexer]

        return monitoring_data

    def _apply_filters(self, monitoring_meta):
        _monitoring_meta = monitoring_meta.copy()
        _monitoring_meta.reset_index(inplace=True)

        for filter in self.filters:
            _monitoring_meta = filter.apply(_monitoring_meta, self.test_dataset)

        _monitoring_meta = _monitoring_meta.sample(frac=1)

        return _monitoring_meta

    def _get_monitoring_output(self, epoch):
        monitoring_meta = self._read_monitoring_meta()

        return monitoring_meta, monitoring_data

    def _read_monitoring_meta(self):
        return pd.read_csv(self._get_meta_file_path())

    def _read_monitoring_data(self, epoch):
        f = h5.File(
            self._get_data_file_path(epoch),
            "r",
        )

        monitoring_data = {}
        for key in self.metric.monitored_params:
            monitoring_data[key] = np.array(f[key])

        f.close()

        return monitoring_data

    def _add_tester(self, training_model_ctor, monitoring_model_ctor):
        self.tester = KFoldTester(
            self.exp_name,
            training_ctor=training_model_ctor,
            monitoring_ctor=monitoring_model_ctor,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            split=self.split,
            epochs=self.epochs,
            monitored_params=self.metric.monitored_params,
            method_params=self.method_params,
        )

    def _are_monitoring_files_present(self):
        return self._are_data_files_present() and self._is_meta_file_present()

    def _are_data_files_present(self):
        data_file_existances = []

        for epoch in self.epochs:
            data_file_existances.append(exists(self._get_data_file_path(epoch)))

        return np.array(data_file_existances).all()

    def _is_meta_file_present(self):
        return exists(self._get_meta_file_path())

    def _get_data_file_path(self, epoch):
        data_file_path = get_monitoring_data_file_path(
            self.exp_name,
            self.training_model_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
            epoch,
            self.metric.monitored_params,
        )

        return data_file_path

    def _get_meta_file_path(self):
        meta_file_path = get_monitoring_meta_file_path(
            self.exp_name,
            self.training_model_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )

        return meta_file_path
