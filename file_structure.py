from config import HOME_DIR, TRAINING_MODELS_DIR, MONITORING_DIR
from os.path import join
import tensorflow as tf
from os import makedirs
import h5py
from kfold_environment import KFoldEnvironment


class Experiment:
    def __init__(
        self,
        exp_name,
        training_model_ctor,
        monitoring_model_ctor,
        train_datasets,
        test_datasets,
        epochs,
        splits,
        method_params,
        monitored_param_list,
    ):
        self._exp_name = exp_name
        self._training_model_ctor = training_model_ctor
        self._monitoring_model_ctor = monitoring_model_ctor
        self._training_model_name = training_model_ctor().name
        self._monitoring_model_name = monitoring_model_ctor().name
        self._train_datasets = train_datasets
        self._test_datasets = test_datasets
        self._epochs = epochs
        self._splits = splits
        self._method_params = method_params
        self._monitored_param_list = monitored_param_list

    def get_traces_file_path(self, train_dataset, test_dataset, split):
        return f"{MONITORING_DIR}/training_{train_dataset}/testing_{test_dataset}/\
            {self._monitoring_model_name}/split{split}/traces.csv"

    def get_monitoring_file_path(self, train_dataset, test_dataset, split):
        return f"{MONITORING_DIR}/training_{train_dataset}/testing_{test_dataset}/\
            {self._monitoring_model_name}/split{self._split_idx}/epoch{self._epoch}_monitoredparams{self._monitored_params}.hdf5"

    def get_monitoring_file_dir(self):
        return f"{MONITORING_DIR}/training_{self._train_dataset}/testing_{self._test_dataset}/\
            {self._monitoring_model_name}/split{self._split_idx}"

    def get_training_model_path(self):
        return f"{TRAINING_MODELS_DIR}/{self.training_model_name}/{self._train_dataset}/split{self._split}/ep{self._epoch}.h5"

    def _init_training_model(self, model):
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
        training_model = self._init_training_model(self.training_ctor())

        if self._epoch is not None:
            training_model.load_weights(self._get_training_model_path())

        monitoring_model = self.monitor_ctor(
            training_model,
            monitor_params=self._monitored_params,
            method_params=self._method_params,
        )
        return monitoring_model

    def _create_test_environment(self):
        # Form test environment.
        test_kenv = KFoldEnvironment(
            dataset=self._test_dataset,
            subsampling_factor=1.0,
        )

        return test_kenv

    def _save_to_traces_file(self, df_traces):
        df_traces.to_csv(self._get_traces_file_path(self._exp_name, self._split))

    def _save_to_monitoring_file(self, monitoring_result):
        with h5py.File(
            self.get_monitoring_file_path(),
            "w",
        ) as f:
            for key in monitoring_result.keys():
                f.create_dataset(key, data=monitoring_result[key])

    def run(
        self,
    ):
        makedirs(self.get_monitoring_file_dir(), exist_ok=True)
        __, __, __, predict_gen = self.test_environment.get_generators(self._split)

        monitoring_model = self._create_monitoring_model()
        monitoring_result = monitoring_model.predict(predict_gen)

        __, __, df_traces = self.test_environment.get_split_metadata(self._split)

        self._save_to_traces_file(df_traces)
        self._save_to_monitoring_file(monitoring_result)
