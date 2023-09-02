from config import TRAINED_MODELS_DIR, HOME_DIR, MONITORING_DIR

from os import makedirs
import h5py
import tensorflow as tf


class KFoldTester:
    def __init__(
        self,
        exp_name,
        training_ctor,
        monitor_ctor,
        train_environment,
        test_environment,
        split,
        epoch,
        monitored_params=[],
        method_params={},
    ):
        self.exp_name = exp_name
        self.training_ctor = training_ctor
        self.monitor_ctor = monitor_ctor
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.split = split
        self.epoch = epoch
        self.monitored_params = monitored_params
        self.method_params = method_params

        self.training_model_name = training_ctor().name
        self.monitoring_model_name = monitor_ctor().name
        self.batch_size = train_environment.batch_size

    def test(
        self,
    ):
        makedirs(self._get_results_dir(self.exp_name, split=self.split), exist_ok=True)
        __, __, __, predict_gen = self.test_environment.get_generators(self.split)

        monitoring_model = self._load_monitoring_model(
            self.split, self.monitored_params, self.method_params
        )

        monitoring_dict = monitoring_model.predict(predict_gen)
        __, __, metadata = self.test_environment.get_split_metadata(self.split)

        self._save_data_file(monitoring_dict)
        self._save_meta_file(metadata)

    def get_monitoring_meta_file_path(self):
        return "{}/{}/training_{}/testing_{}/{}/split{}/meta.csv".format(
            MONITORING_DIR,
            self.exp_name,
            self.train_environment.dataset,
            self.test_environment.dataset,
            self.monitoring_model_name,
            self.split,
        )

    def get_monitoring_data_file_path(self):
        return "{}/{}/training_{}/testing_{}/{}/split{}/epoch{}_monitoredparams{}.hdf5".format(
            MONITORING_DIR,
            self.exp_name,
            self.train_environment.dataset,
            self.test_environment.dataset,
            self.monitoring_model_name,
            self.split,
            self.epoch,
            self.monitored_params,
        )

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

        model(
            tf.random.normal(
                shape=(self.batch_size, model.N_TIMESTEPS, model.N_CHANNELS)
            )
        )

        return model

    def _create_monitoring_model(self):
        training_model = self._create_training_model

        if self.epoch is not None:
            training_model.load_weights(
                self._get_training_model_path(self.split, epoch=self.epoch)
            )

        monitoring_model = self.monitor_ctor(
            training_model,
            monitor_params=self.monitored_params,
            method_params=self.method_params,
        )
        return monitoring_model
