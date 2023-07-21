from config import (
    TRAINED_MODELS_DIR,
    HOME_DIR,
)

from os.path import join
from os import makedirs
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn


class MyCallback(tf.keras.callbacks.Callback):
    """
    A callback that saves the model weights at the end of each epoch.

    Parameters
    ----------
    model : tf.keras.Model
        The model.
    dir : str
        The directory where the model weights are saved.
    """

    def __init__(self, model, dir):
        self.model = model
        self.dir = dir

    def on_epoch_end(self, epoch, logs):
        self.model.save_weights(join(self.dir, "ep{}.h5".format(epoch)))


class KFoldTester:
    """
    Tests a detector model.

    Parameters
    ----------
    train_environment : KFoldEnvironment
        The environment used for training.
    test_environment : KFoldEnvironment
        The environment used for testing.
    detector_ctor : function
        The constructor of the detector model.
    monitor_ctor : function
        The constructor of the monitoring model.
    detector_models_home_dir : str
        The directory where the detector models are saved.
    results_home_dir : str
        The directory where the results are saved.
    """

    def __init__(
        self,
        train_environment,
        test_environment,
        detector_ctor,
        monitor_ctor,
        detector_models_home_dir=TRAINED_MODELS_DIR,
        home_dir=HOME_DIR,
    ):
        assert train_environment.batch_size == test_environment.batch_size
        assert train_environment.n_splits == test_environment.n_splits

        self.detector_models_home_dir = detector_models_home_dir
        self.detector_ctor = detector_ctor
        self.monitor_ctor = monitor_ctor
        self.batch_size = train_environment.batch_size
        self.train_dataset = train_environment.dataset
        self.test_dataset = test_environment.dataset
        self.home_dir = home_dir
        self.train_environment = train_environment
        self.test_environment = test_environment

    def _load_detector_model(self, model, model_path):
        """
        Loads a model from a file.

        Parameters
        ----------
        model : tf.keras.Model
            The model.
        model_path : str
            The path of the file.

        Returns
        -------
        tf.keras.Model
            The loaded model.
        """

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[],
        )

        model(
            tf.random.normal(
                shape=(self.batch_size, model.N_TIMESTEPS, model.N_CHANNELS)
            )
        )
        model.load_weights(model_path)
        return model

    def _get_detector_model_dir(self, split):
        """
        Returns the directory where the detector model is saved for a split.

        Parameters
        ----------
        split : int
            The split index.

        Returns
        -------
        str
            The directory where the detector model is saved for a split.
        """
        return join(
            self.detector_models_home_dir,
            self.train_dataset,
            self.detector_ctor().name,
            "split{}".format(split),
        )

    def _get_results_dir(self, exp_name, split):
        """
        Returns the directory where results are saved for a split.

        Parameters
        ----------
        exp_name : str
            The experiment name.
        split : int
            The split index.

        Returns
        -------
        str
            The directory where results are saved for a split.
        """
        monitordir = join(
            self.home_dir,
            "results",
            exp_name,
            "training_{}".format(self.train_dataset),
            "testing_{}".format(self.test_dataset),
            self.monitor_ctor().name,
            "split{}".format(split),
        )
        return monitordir

    def _get_traces_csv_name(self, n_batches=None):
        """
        Returns the name of the traces file.

        Parameters
        ----------
        n_batches: int
            The number of batches.

        Returns
        -------
        str
            The name of the traces file.
        """
        if n_batches is not None:
            fname = "traces_nbatches{}.csv".format(
                n_batches,
            )
        else:
            fname = "traces.csv"

        return fname

    def _get_monitor_hdf5_name(self, epoch, monitored_params, n_batches=None):
        """
        Returns the name of the monitor file.

        Parameters
        ----------
        epoch : int
            The epoch index.
        monitored_params : list of str
            The parameters which are monitored.
        n_batches: int
            The number of batches.

        Returns
        -------
        str
            The name of the monitor file.
        """
        if n_batches is not None:
            fname = "epoch{}_monitoredparams{}_nbatches{}.hdf5".format(
                epoch,
                str(monitored_params),
                n_batches,
            )
        else:
            fname = "epoch{}_monitoredparams{}.hdf5".format(
                epoch,
                str(monitored_params),
            )

        return fname

    def _get_monitor_hdf5_path(
        self, exp_name, epoch, split, monitor_params, n_batches=None
    ):
        """
        Returns the path of the monitor file.

        Parameters
        ----------
        exp_name : str
            The experiment name.
        epoch : int
            The epoch index.
        split : int
            The split index.
        monitor_params : list of str
            The parameters which are monitored.
        n_batches: int
            The number of batches.

        Returns
        -------
        str
            The path of the monitor file.
        """
        monitor_path = join(
            self._get_results_dir(exp_name, split),
            self._get_monitor_hdf5_name(epoch, monitor_params, n_batches=n_batches),
        )
        return monitor_path

    def _get_traces_csv_path(self, exp_name, split, n_batches=None):
        """
        Returns the path of the traces file.

        Parameters
        ----------
        exp_name : str
            The experiment name.
        split : int
            The split index.
        n_batches: int
            The number of batches.

        Returns
        -------
        str
            The path of the traces file.
        """

        return join(
            self._get_results_dir(exp_name, split), self._get_traces_csv_name(n_batches)
        )

    def _get_detector_model_path(self, split, epoch):
        """
        Returns the path of the detector model.

        Parameters
        ----------
        split : int
            The split index.

        Returns
        -------
        str
            The path of the detector model.
        """

        return join(self._get_detector_model_dir(split), "ep{}.h5".format(epoch))

    def _load_monitoring_model(self, split_idx, monitor_params, method_params):
        """
        Loads a monitoring model.

        Parameters
        ----------
        split_idx : int
            The split index.
        monitor_params : dict
            The parameters of the monitoring model.
        method_params : dict
            The parameters of the detector model.

        Returns
        -------
        tf.keras.Model
            The loaded monitoring model.
        """

        model = self._load_detector_model(
            self.detector_ctor(),
            self._get_detector_model_path(split_idx, epoch=method_params["epoch"]),
        )

        monitoring_model = self.monitor_ctor(
            model,
            monitor_params=monitor_params,
            method_params=method_params,
        )
        return monitoring_model

    def _concatenate_monitoring_results(self, results):
        """
        Concatenates monitoring results.

        Parameters
        ----------
        results : list of dict
            The monitoring results.

        Returns
        -------
        dict
            The concatenated monitoring results.
        """

        concatenated_results = {}
        for key in results[0].keys():
            concatenated_results[key] = np.concatenate(
                [result[key] for result in results], axis=0
            )

        return concatenated_results

    def test_samples(
        self,
        exp_name,
        splits=None,
        method_params={"epoch": []},
        monitor_params=[],
        n_batches_per_split=None,
    ):
        """
        Tests a method.

        Parameters
        ----------
        exp_name : str
            The experiment name.
        splits : list of int
            The splits to test.
        method_params : dict
            The parameters of the method.
        monitor_params : list
            What to monitor.
        n_batches_per_split : int
            The number of batches to test per split.

        Returns
        -------
        None
        """

        if splits is None:
            splits = range(self.splits)

        for split_idx in splits:
            # Create results directory.
            makedirs(self._get_results_dir(exp_name, split=split_idx), exist_ok=True)

            __, __, test_gen, predict_gen = self.test_environment.get_generators(
                split_idx
            )

            # Loads the model which is used for monitoring the detector models intermediate layers.
            monitoring_model = self._load_monitoring_model(
                split_idx, monitor_params, method_params
            )

            num_samples = test_gen.__len__()
            dfs = []
            monitoring_results = []

            for i in range(n_batches_per_split):
                # Select a random batch.
                batch_idx = rn.randint(0, num_samples - 1)

                # Get batch.
                x, y = test_gen.__getitem__(batch_idx)

                # Return monitoring result for the batch.
                monitoring_result = monitoring_model.predict(x)

                # Get dataframe corresponding to the batch.
                batch_df = self.test_environment.get_batch_metadata(
                    split_idx, "test", batch_idx
                )

                dfs.append(batch_df)
                monitoring_results.append(monitoring_result)

            # Save test dataframe to results folder.
            pd.concat(dfs).to_csv(
                self._get_traces_csv_path(exp_name, split_idx, n_batches_per_split)
            )

            # Concatenate monitoring results.
            monitoring_result = self._concatenate_monitoring_results(monitoring_results)

            # Create monitor hdf5 file in order to store the results.
            with h5py.File(
                self._get_monitor_hdf5_path(
                    exp_name,
                    method_params["epoch"],
                    split_idx,
                    monitor_params=monitor_params,
                ),
                "w",
            ) as f:
                for key in monitoring_result.keys():
                    f.create_dataset(key, data=monitoring_result[key])

    def test(
        self,
        exp_name,
        splits=None,
        method_params={"epoch": []},
        monitor_params=[],
    ):
        """
        Tests a method.

        Parameters
        ----------
        exp_name : str
            The experiment name.
        splits : list of int
            The splits to test.
        method_params : dict
            The parameters of the method.
        monitor_params : list
            Params to monitor.

        Returns
        -------
        None
        """

        if splits is None:
            splits = range(self.splits)

        for split_idx in splits:
            # Create results directory.
            makedirs(self._get_results_dir(exp_name, split=split_idx), exist_ok=True)

            __, __, __, predict_gen = self.test_environment.get_generators(split_idx)

            # Loads the model which is used for monitoring the detector models intermediate layers.
            monitoring_model = self._load_monitoring_model(
                split_idx, monitor_params, method_params
            )

            # Return monitoring results.
            monitoring_result = monitoring_model.predict(predict_gen)

            # Get test dataframe used through monitoring process.
            __, __, test_df = self.test_environment.get_split_metadata(split_idx)

            # Save test dataframe to results folder.
            test_df.to_csv(self._get_traces_csv_path(exp_name, split_idx))

            # Create monitor hdf5 file in order to store the results.
            with h5py.File(
                self._get_monitor_hdf5_path(
                    exp_name,
                    method_params["epoch"],
                    split_idx,
                    monitor_params=monitor_params,
                ),
                "w",
            ) as f:
                for key in monitoring_result.keys():
                    f.create_dataset(key, data=monitoring_result[key])
