import pandas as pd
import numpy as np
import h5py as h5
from os.path import exists
from kfold_environment import KFoldEnvironment
from kfold_tester import KFoldTester
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
        df_traces = df_traces.copy()
        df_eq_traces, df_no_traces = self._split_eq_and_noise_traces()

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
        df_traces = df_traces.copy()
        df_eq_traces, df_no_traces = self._split_eq_and_noise_traces()

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


class Evaluate:
    def __init__(
        self,
        exp_name,
        training_model_ctor,
        monitoring_model_ctor,
        train_dataset,
        test_dataset,
        epoch,
        split_idx,
        method_params,
        monitored_param_list,
        filters,
        metric,
    ):
        self._exp_name = exp_name
        self._epoch = epoch
        self._split_idx = split_idx
        self._method_params = method_params
        self._monitored_param_list = monitored_param_list
        self._filters = filters
        self._metric = metric

        self._add_tester(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            training_model_ctor=training_model_ctor,
            monitoring_model_ctor=monitoring_model_ctor,
        )

    def get_roc_vectors(
        self,
    ):
        monitoring_meta, monitoring_data = self._get_monitoring_output()
        monitoring_meta, monitoring_data = self._apply_filters(
            monitoring_meta, monitoring_data
        )

        metric = self._metric(monitoring_data)
        labels = monitoring_meta["label"] == "eq"

        fpr, tpr, thresholds = roc_curve(labels, metric)

        return tpr, fpr, thresholds

    def _apply_filters(self, monitoring_meta, monitoring_data):
        _monitoring_meta = monitoring_meta.copy()
        _monitoring_meta.reset_index(inplace=True)

        for filter in self._filters:
            _monitoring_meta = filter.apply(_monitoring_meta)

        _monitoring_meta = _monitoring_meta.sample(frac=1)

        indexer = np.array(_monitoring_meta.index)
        for key in self._monitored_param_list:
            monitoring_data[key] = monitoring_data[key][indexer]

        return _monitoring_meta, monitoring_data

    def _get_monitoring_output(self):
        if not self._are_monitoring_files_present():
            self._monitor()

        monitoring_meta = self._read_monitoring_meta()
        monitoring_data = self._read_monitoring_data()

        return monitoring_meta, monitoring_data

    def _are_monitoring_files_present(self):
        return exists(self.tester.get_monitoring_data_file_path()) and exists(
            self.tester.get_monitoring_meta_file_path()
        )

    def _read_monitoring_meta(self):
        return pd.read_csv(self.tester.get_monitoring_meta_file_path())

    def _read_monitoring_data(
        self,
    ):
        f = h5.File(
            self.tester.get_monitoring_data_file_path(),
            "r",
        )

        monitoring_data = {}
        for key in self._monitored_param_list:
            monitoring_data[key] = np.array(f[key])

        f.close()

        return monitoring_data

    def _monitor(self):
        self.tester.test(
            exp_name=self._exp_name,
            splits=self._split_idx,
            monitored_params=self._monitored_param_list,
            method_params=self._method_params,
        )

    def _add_tester(
        self, train_dataset, test_dataset, training_model_ctor, monitoring_model_ctor
    ):
        train_kenv = KFoldEnvironment(
            dataset=train_dataset,
        )

        test_kenv = KFoldEnvironment(
            dataset=test_dataset,
        )

        self.tester = KFoldTester(
            train_environment=train_kenv,
            test_environment=test_kenv,
            detector_ctor=training_model_ctor,
            monitor_ctor=monitoring_model_ctor,
        )
