from directory import *
from config import BATCH_SIZE

from os import makedirs
import h5py
import tensorflow as tf
from kfold_environment import KFoldEnvironment


class KFoldTester:
    def __init__(
        self,
        exp_name,
        training_ctor,
        monitor_ctor,
        train_dataset,
        test_dataset,
        split,
        epoch,
        monitored_params=[],
        method_params={},
    ):
        self.exp_name = exp_name
        self.training_ctor = training_ctor
        self.monitor_ctor = monitor_ctor
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split = split
        self.epoch = epoch
        self.monitored_params = monitored_params
        self.method_params = method_params

        self.training_model_name = training_ctor().name
        self.monitoring_model_name = monitor_ctor().name
        self._add_test_environment()

    def test(
        self,
    ):
        monitoring_output_dir = self._get_monitoring_output_dir()

        makedirs(monitoring_output_dir, exist_ok=True)

        __, __, __, predict_gen = self.test_environment.get_generators(self.split)

        monitoring_model = self._create_monitoring_model()

        monitoring_dict = monitoring_model.predict(predict_gen)
        __, __, metadata = self.test_environment.get_split_metadata(self.split)

        self._save_data_file(monitoring_dict)
        self._save_meta_file(metadata)

    def _save_data_file(self, monitoring_dict):
        with h5py.File(
            self.get_monitoring_data_file_path(),
            "w",
        ) as f:
            for key in monitoring_dict.keys():
                f.create_dataset(key, data=monitoring_dict[key])

    def _save_meta_file(self, metadata):
        metadata.to_csv(self.get_monitoring_meta_file_path())

    def _create_training_model(self):
        model = self.training_ctor()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[],
        )

        model(tf.random.normal(shape=(BATCH_SIZE, model.N_TIMESTEPS, model.N_CHANNELS)))

        return model

    def _create_monitoring_model(self):
        training_model = self._create_training_model()

        if self.epoch is not None:
            training_model.load_weights(
                get_checkpoint_path(
                    self.training_model_name, self.train_dataset, self.split, self.epoch
                )
            )

        monitoring_model = self.monitor_ctor(
            training_model,
            monitor_params=self.monitored_params,
            method_params=self.method_params,
        )
        return monitoring_model

    def _add_test_environment(self):
        self.test_environment = KFoldEnvironment(self.train_dataset)

    def _get_monitoring_output_dir(self):
        return get_monitoring_output_dir(
            self.exp_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )
