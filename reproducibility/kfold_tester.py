import pandas as pd
import tensorflow as tf
from os import makedirs
from seismic_purifier import BATCH_SIZE
from kfold_environment import KFoldEnvironment
import numpy as np
from directory import *

class KFoldTester:
    def __init__(
        self,
        exp_name,
        representation_learning_model_class,
        classifier_model_class,
        train_dataset,
        test_dataset,
        split,
        epochs,
        method_params={},
    ):
        self.exp_name = exp_name
        self.representation_learning_model_class = representation_learning_model_class
        self.classifer_model_class = classifier_model_class
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split = split
        self.epochs = epochs
        self.method_params = method_params

        self.representation_learning_model_name = representation_learning_model_class().name
        self.classifier_model_name = classifier_model_class().name
        self._add_test_environment()

    def test(
        self,
    ):
        exp_results_dir = self._get_exp_results_dir()

        makedirs(exp_results_dir, exist_ok=True)
        __, __, __, predict_gen = self.test_environment.get_generators(self.split)

        for epoch in self.epochs:
            classifier_model = self._create_classifier_model(epoch)
            scores = self._predict(classifier_model, predict_gen)
            self._save_score_file(scores, epoch)

        __, __, metadata = self.test_environment.get_split_metadata(self.split)
        self._save_meta_file(metadata)

    def _predict(self, classifier_model, predict_gen):
        outputs = []
        for i in range(predict_gen.__len__()):
            x = predict_gen.__getitem__(i)
            y = classifier_model(x)
            outputs.append(y)
        
        output = np.concatenate(outputs, axis=0)
        return output
    
    def _save_score_file(self, scores, epoch):
        score_file_path = get_exp_results_score_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
            epoch
        )

        df_score = pd.DataFrame({"eq_probabilities":scores})
        df_score.to_csv(score_file_path)

    def _save_meta_file(self, metadata):
        meta_file_path = get_exp_results_meta_file_path(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )

        metadata.to_csv(meta_file_path)

    def _create_representation_learning_model(self):
        model = self.representation_learning_model_class()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[],
        )

        model(tf.random.normal(shape=(BATCH_SIZE, model.N_TIMESTEPS, model.N_CHANNELS)))

        return model

    def _create_classifier_model(self, epoch):
        if self.representation_learning_model_class is not None:
            representation_learning_model = self._create_representation_learning_model()

            representation_learning_model.load_weights(
                get_checkpoint_path(
                    self.exp_name,
                    self.representation_learning_model_name,
                    self.train_dataset,
                    self.split,
                    epoch,
                )
            )
        else:
            representation_learning_model = None

        classifier_model = self.classifer_model_class(
            representation_learning_model,
            method_params=self.method_params,
        )
        return classifier_model

    def _add_test_environment(self):
        self.test_environment = KFoldEnvironment(self.test_dataset)

    def _get_exp_results_dir(self):
        return get_exp_results_dir(
            self.exp_name,
            self.representation_learning_model_name,
            self.classifier_model_name,
            self.train_dataset,
            self.test_dataset,
            self.split,
        )