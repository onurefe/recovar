from config import BATCH_SIZE
from directory import TRAINED_MODELS_DIR
from kfold_environment import KFoldEnvironment
from os.path import join
from os import makedirs
import pandas as pd
import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dir):
        self.model = model
        self.dir = dir

    def on_epoch_end(self, epoch, logs):
        self.model.save_weights(join(self.dir, "ep{}.h5".format(epoch)))


class KfoldTrainer:
    def __init__(
        self,
        model_ctor,
        dataset,
        epochs,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        self.model_ctor = model_ctor
        self.model_name = model_ctor().get_config["name"]
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    
    def _get_checkpoint_dir(self, split):
        return join(TRAINED_MODELS_DIR, self.dataset, self.model_name, 
                    "split{}".format(split))

    def _get_history_csv_path(self, split):
        return join(TRAINED_MODELS_DIR, self.dataset, self.model_name, 
                    "split{}".format(split), "history.csv")

    def _get_checkpoint_path(self, split, epoch):
        return join(TRAINED_MODELS_DIR, self.dataset, self.model_name, 
                    "split{}".format(split), "ep{}.h5".format(epoch))

    def _create_model(self):
        model = self.model_ctor()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
            ),
            metrics=[],
        )

    def _train_model(self, model, split, train_gen, validation_gen):
        fit_result = model.fit(
            train_gen,
            validation_gen,
            epochs=self.epochs,
            batch_size=BATCH_SIZE,
            callbacks=[
                MyCallback(
                    model,
                    self._get_checkpoint_dir(split),
                )
            ],
            shuffle=False,
        )

        return fit_result

    def _save_history(self, split, fit_result):
        with open(
            self._get_history_csv_path(split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(fit_result.history)
            hist_df.to_csv(f)

    def train(
        self,
        split,
    ):
        kfold_env = KFoldEnvironment(
            dataset=self.dataset,
            subsampling_factor=1.0,
        )

        (
            train_gen,
            validation_gen,
            test_gen,
            predict_gen,
        ) = kfold_env.get_generators(split)

        makedirs(
            self._get_checkpoint_dir(split),
            exist_ok=True,
        )

        model = self._create_model()

        fit_result = self._train_model(
            model=model, split=split, train_gen=train_gen, validation_gen=validation_gen
        )

        self._save_history(split, fit_result)

    def train_for_splits(self, splits):
        for split in splits:
            self.train(split)
