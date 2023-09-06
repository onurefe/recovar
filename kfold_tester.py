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
        monitoring_ctor,
        train_dataset,
        test_dataset,
        split,
        epochs,
        monitored_params=[],
        method_params={},
    ):
        self.exp_name = exp_name
        self.training_ctor = training_ctor
        self.monitoring_ctor = monitoring_ctor
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split = split
        self.epochs = epochs
        self.monitored_params = monitored_params
        self.method_params = method_params

        self.training_model_name = training_ctor().name
        self.monitoring_model_name = monitoring_ctor().name
        self._add_test_environment()

    def test(
        self,
    ):
        monitoring_output_dir = self._get_monitoring_output_dir()

        makedirs(monitoring_output_dir, exist_ok=True)
        __, __, __, predict_gen = self.test_environment.get_generators(self.split)

        for epoch in self.epochs:
            monitoring_model = self._create_monitoring_model(epoch)
            monitoring_dict = monitoring_model.predict(predict_gen)
            self._save_data_file(monitoring_dict, epoch)

        __, __, metadata = self.test_environment.get_split_metadata(self.split)
        self._save_meta_file(metadata)

    def _save_data_file(self, monitoring_dict, epoch):
        data_file_path = get_monitoring_data_file_path(
            self.exp_name,
            self.training_model_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
            epoch,
            self.monitored_params,
        )

        with h5py.File(data_file_path, "w") as f:
            for key in monitoring_dict.keys():
                f.create_dataset(key, data=monitoring_dict[key])

    def _save_meta_file(self, metadata):
        meta_file_path = get_monitoring_meta_file_path(
            self.exp_name,
            self.training_model_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )

        metadata.to_csv(meta_file_path)

    def _create_training_model(self):
        model = self.training_ctor()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[],
        )

        model(tf.random.normal(shape=(BATCH_SIZE, model.N_TIMESTEPS, model.N_CHANNELS)))

        return model

    def _create_monitoring_model(self, epoch):
        if self.training_ctor is not None:
            training_model = self._create_training_model()

            training_model.load_weights(
                get_checkpoint_path(
                    self.exp_name,
                    self.training_model_name,
                    self.train_dataset,
                    self.split,
                    epoch,
                )
            )
        else:
            training_model = None

        monitoring_model = self.monitoring_ctor(
            training_model,
            monitored_params=self.monitored_params,
            method_params=self.method_params,
        )
        return monitoring_model

    def _add_test_environment(self):
        self.test_environment = KFoldEnvironment(self.train_dataset)

    def _get_monitoring_output_dir(self):
        return get_monitoring_output_dir(
            self.exp_name,
            self.training_model_name,
            self.monitoring_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )
