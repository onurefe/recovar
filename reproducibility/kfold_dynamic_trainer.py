import numpy as np
import pandas as pd
import tensorflow as tf
from recovar import BATCH_SIZE, ClassifierMultipleAutoencoder
from recovar.utils import demean, l2_normalize
from kfold_trainer import KfoldTrainer
from kfold_environment import KFoldEnvironment
from directory import *
from os import makedirs

class KfoldDynamicTrainer(KfoldTrainer):
    def __init__(
        self,
        exp_name,
        model_class,
        dataset,
        split,
        epochs,
        batch_multiplier=10,  
        keep_top_batches=2,        
        apply_resampling=False,
        resampling_eq_ratio=0.5,
        resample_while_keeping_total_waveforms_fixed=False,
        learning_rate=1e-4,
        epsilon=1e-7,
        beta_1=0.99,
        beta_2=0.999,
    ):
        super().__init__(
            exp_name=exp_name,
            model_class=model_class,
            dataset=dataset,
            split=split,
            epochs=epochs,
            apply_resampling=apply_resampling,
            resampling_eq_ratio=resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=resample_while_keeping_total_waveforms_fixed,
            learning_rate=learning_rate,
            epsilon=epsilon,
            beta_1=beta_1,
            beta_2=beta_2,
        )

        self.batch_multiplier = batch_multiplier
        self.keep_top_batches = keep_top_batches

    def train(self):
        kfold_env = KFoldEnvironment(
            dataset=self.dataset,
            apply_resampling=self.apply_resampling,
            resample_eq_ratio=self.resampling_eq_ratio,
            resample_while_keeping_total_waveforms_fixed=self.resample_while_keeping_total_waveforms_fixed
        )

        train_gen, validation_gen, _, _ = kfold_env.get_generators(self.split)

        makedirs(
            get_checkpoint_dir(
                self.exp_name, self.model_name, self.dataset, self.split
            ),
            exist_ok=True,
        )

        model = self._create_model()
        model.build(input_shape=(None, 3000, 3))
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )

        classifier = ClassifierMultipleAutoencoder(model=model)

        history = {'loss': [], 'val_loss': []}

        for epoch in range(self.epochs):

            epoch_loss = self._train_one_epoch(
                model, optimizer, train_gen, classifier, epoch
            )
            val_loss = self._validate(model, validation_gen)

            checkpoint_path = get_checkpoint_path(
                self.exp_name, self.model_name, self.dataset, self.split, epoch
            )
            model.save_weights(checkpoint_path)

            history['loss'].append(float(epoch_loss))
            history['val_loss'].append(float(val_loss))

            print(f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self._save_history(self.split, history)

    def _train_one_epoch(self, model, optimizer, train_gen, classifier, epoch):
        n_batches = len(train_gen)
        epoch_losses = []

        #Start with batch_multiplier, decrease by 1 each epoch until keep_top_batches
        effective_batch_multiplier = max(
            self.keep_top_batches,
            self.batch_multiplier - epoch
        )

        print(f"Epoch {epoch}: Using batch_multiplier = {effective_batch_multiplier} "
              f"(keeping top {self.keep_top_batches} batches)")

        n_super_batches = n_batches // effective_batch_multiplier

        for super_batch_idx in range(n_super_batches):
            x_batches = []
            y_batches = []

            for mult_idx in range(effective_batch_multiplier):
                batch_idx = super_batch_idx * effective_batch_multiplier + mult_idx
                x_batch, y_batch = train_gen[batch_idx]
                x_batches.append(x_batch)
                y_batches.append(y_batch)

            x_all = np.concatenate(x_batches, axis=0) 
            y_all = np.concatenate(y_batches, axis=0)

            scores = classifier(x_all, training=False)

            n_keep = self.keep_top_batches * BATCH_SIZE
            top_indices = np.argsort(scores)[::-1][:n_keep]  #Top n_keep samples

            x_selected = x_all[top_indices]
            y_selected = y_all[top_indices]

            avg_score = scores.mean()
            top_score = scores[top_indices].mean()

            with tf.GradientTape() as tape:
                f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = model(x_selected, training=True)

                per_sample_recon_loss = (
                    self._per_sample_l2_distance(x_selected, y1) +
                    self._per_sample_l2_distance(x_selected, y2) +
                    self._per_sample_l2_distance(x_selected, y3) +
                    self._per_sample_l2_distance(x_selected, y4) +
                    self._per_sample_l2_distance(x_selected, y5)
                ) / 5.0

                per_sample_ensemble_loss = (
                    self._per_sample_ensemble_distance(f1, f2) +
                    self._per_sample_ensemble_distance(f1, f3) +
                    self._per_sample_ensemble_distance(f2, f3) +
                    self._per_sample_ensemble_distance(f1, f4) +
                    self._per_sample_ensemble_distance(f2, f4) +
                    self._per_sample_ensemble_distance(f3, f4) +
                    self._per_sample_ensemble_distance(f1, f5) +
                    self._per_sample_ensemble_distance(f2, f5) +
                    self._per_sample_ensemble_distance(f3, f5) +
                    self._per_sample_ensemble_distance(f4, f5)
                ) / 10.0

                per_sample_total_loss = per_sample_recon_loss + per_sample_ensemble_loss
                loss = tf.reduce_mean(per_sample_total_loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_losses.append(float(loss))

            if (super_batch_idx + 1) % 10 == 0:

                print(f"  Super-batch {super_batch_idx+1}/{n_super_batches}, "
                        f"Loss: {loss:.4f}, "
                        f"Avg score: {avg_score:.4f}, Top score: {top_score:.4f}, "
                        f"Batch multiplier: {effective_batch_multiplier}")


        return np.mean(epoch_losses)

    def _validate(self, model, validation_gen):
        n_batches = len(validation_gen)
        val_losses = []

        for batch_idx in range(n_batches):
            x, y = validation_gen[batch_idx]

            f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = model(x, training=False)

            per_sample_recon_loss = (
                self._per_sample_l2_distance(x, y1) +
                self._per_sample_l2_distance(x, y2) +
                self._per_sample_l2_distance(x, y3) +
                self._per_sample_l2_distance(x, y4) +
                self._per_sample_l2_distance(x, y5)
            ) / 5.0

            per_sample_ensemble_loss = (
                self._per_sample_ensemble_distance(f1, f2) +
                self._per_sample_ensemble_distance(f1, f3) +
                self._per_sample_ensemble_distance(f2, f3) +
                self._per_sample_ensemble_distance(f1, f4) +
                self._per_sample_ensemble_distance(f2, f4) +
                self._per_sample_ensemble_distance(f3, f4) +
                self._per_sample_ensemble_distance(f1, f5) +
                self._per_sample_ensemble_distance(f2, f5) +
                self._per_sample_ensemble_distance(f3, f5) +
                self._per_sample_ensemble_distance(f4, f5)
            ) / 10.0

            per_sample_total_loss = per_sample_recon_loss + per_sample_ensemble_loss
            loss = tf.reduce_mean(per_sample_total_loss)

            val_losses.append(float(loss))

        return np.mean(val_losses)

    def _per_sample_l2_distance(self, x, y):
        x_demeaned = demean(x)
        y_demeaned = demean(y)

        distance = tf.sqrt(tf.reduce_mean(tf.square(x_demeaned - y_demeaned), axis=[1, 2]))
        return distance

    def _per_sample_ensemble_distance(self, f1, f2):
        f1_normalized = demean(f1, axis=1)
        f2_normalized = demean(f2, axis=1)

        f1_normalized = l2_normalize(f1_normalized, axis=1)
        f2_normalized = l2_normalize(f2_normalized, axis=1)

        distance = tf.sqrt(tf.reduce_mean(tf.square(f1_normalized - f2_normalized), axis=[1, 2]))
        return distance

    def _save_history(self, split, history_dict):
        with open(
            get_history_csv_path(self.exp_name, self.model_name, self.dataset, split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(history_dict)
            hist_df.to_csv(f)
