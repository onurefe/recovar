from config import TRAINED_MODELS_DIR, STEAD_TIME_WINDOW, INSTANCE_TIME_WINDOW

from kfold_environment import KFoldEnvironment

from detector_models import (
    AutocovarianceDetector30s,
    AutocovarianceDetectorDenoising30s,
    AutocovarianceDetectorDenoisingRealisticNoise30s,
    Ensemble5CrossCovarianceDetector30s,
)
from os.path import join
from os import makedirs
import pandas as pd
import tensorflow as tf


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


class Trainer:
    """
    Trains a detector model.

    Parameters
    ----------
    model_ctor : function
        The constructor of the model.
    models_home_dir : str
        The directory where the models are saved.
    """

    def __init__(self, model_ctor, models_home_dir=TRAINED_MODELS_DIR):
        self.model_ctor = model_ctor
        self.model_name = model_ctor().name
        self.models_home_dir = models_home_dir

    def _get_model_dir(self, dataset):
        """
        Returns the directory where the model is saved.
        """

        return join(self.models_home_dir, dataset, self.model_name)

    def _get_split_dir(self, split, dataset):
        """
        Returns the directory where the model of a specific split is saved.

        Parameters
        ----------
        split : int
            The split index.
        """
        return join(self._get_model_dir(dataset), "split{}".format(split))

    def _get_history_csv_path(self, split, dataset):
        """
        Returns the path of the history csv file of a model of specific split.

        Parameters
        ----------
        split : int
            The split index.
        """
        return join(self._get_split_dir(split, dataset), "history.csv")

    def _get_epoch_path(self, split, epoch, dataset):
        """
        Returns the path of the epoch file of a model of specific split.

        Parameters
        ----------
        split : int
            The split index.
        epoch : int
            The epoch index.
        """
        return join(self._get_split_dir(split, dataset), "ep{}.h5".format(epoch))

    def train(
        self,
        dataset,
        epochs=5,
        splits=[],
        starting_epoch=0,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        """
        Trains a detector model.

        Parameters
        ----------
        epochs : int
            The number of epochs.
        starting_split : int
            The starting split index.
        starting_epoch : int
            The starting epoch index.
        learning_rate : float
            The learning rate for the Adam optimizer
        epsilon : float
            The epsilon value for the Adam optimizer
        beta_1 : float
            The beta_1 value for the Adam optimizer
        beta_2 : float
            The beta_2 value for the Adam optimizer
        """
        # Each split would be a dictionary. And would include train, validation and
        # test chunk lists.
        for split_idx in splits:
            kfold_env = KFoldEnvironment(
                dataset=dataset,
                subsampling_factor=1.0,
            )

            (
                train_gen,
                validation_gen,
                test_gen,
                predict_gen,
            ) = kfold_env.get_generators(split_idx)

            model = self.model_ctor()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                ),
                metrics=[],
            )

            makedirs(
                self._get_split_dir(split_idx, dataset),
                exist_ok=True,
            )

            if starting_epoch > 0:
                model(
                    tf.random.normal(
                        (kfold_env.batch_size, model.N_TIMESTEPS, model.N_CHANNELS)
                    )
                )
                model.load_weights(
                    self._get_epoch_path(split_idx, starting_epoch - 1, dataset)
                )

            fit_result = model.fit(
                train_gen,
                validation_data=validation_gen,
                epochs=epochs,
                batch_size=kfold_env.batch_size,
                callbacks=[
                    MyCallback(
                        model,
                        self._get_split_dir(split_idx, dataset),
                    )
                ],
                initial_epoch=starting_epoch,
                shuffle=False,
            )

            starting_epoch = 0

            # Save history.
            with open(
                self._get_history_csv_path(split_idx, dataset),
                "w",
            ) as f:
                hist_df = pd.DataFrame(fit_result.history)
                hist_df.to_csv(f)
