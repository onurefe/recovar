from recovar import BATCH_SIZE
from kfold_environment import KFoldEnvironment
from directory import *
from os import makedirs
import pandas as pd
import tensorflow as tf

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, exp_name, model, train_dataset, split):
        self.exp_name = exp_name
        self.model = model
        self.train_dataset = train_dataset
        self.split = split

    def on_epoch_end(self, epoch, logs):
        checkpoint_path = get_checkpoint_path(
            self.exp_name, self.model.name, self.train_dataset, self.split, epoch
        )
        self.model.save_weights(checkpoint_path)

class KfoldTrainer:
    def __init__(
        self,
        exp_name,
        model_class,
        dataset,
        split,
        epochs,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        self.exp_name = exp_name
        self.model_class = model_class
        self.dataset = dataset
        self.split = split
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.model_name = model_class().name

    def train(
        self,
    ):
        kfold_env = KFoldEnvironment(
            dataset=self.dataset,
        )

        (
            train_gen,
            validation_gen,
            __,
            __,
        ) = kfold_env.get_generators(self.split)

        makedirs(
            get_checkpoint_dir(
                self.exp_name, self.model_name, self.dataset, self.split
            ),
            exist_ok=True,
        )

        model = self._create_model()

        fit_result = self._train_model(
            model=model,
            split=self.split,
            train_gen=train_gen,
            validation_gen=validation_gen,
        )

        self._save_history(self.split, fit_result)

    def _train_model(self, model, split, train_gen, validation_gen):
        checkpoint_callback = CheckpointCallback(
            self.exp_name,
            model,
            self.dataset,
            split,
        )

        fit_result = model.fit(
            train_gen,
            validation_data=validation_gen,
            epochs=self.epochs,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint_callback],
            shuffle=False,
        )

        return fit_result

    def _create_model(self):
        model = self.model_class()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
            ),
            metrics=[],
        )

        return model

    def _save_history(self, split, fit_result):
        with open(
            get_history_csv_path(self.exp_name, self.model_name, self.dataset, split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(fit_result.history)
            hist_df.to_csv(f)