import numpy as np
import pandas as pd
import tensorflow as tf
from recovar import BATCH_SIZE, ClassifierMultipleAutoencoder
from recovar.utils import demean, l2_normalize
from kfold_trainer import KfoldTrainer
from kfold_environment import KFoldEnvironment
from directory import *
from os import makedirs

class KfoldTrainerWithDilation(KfoldTrainer):
    def __init__(
        self,
        exp_name,
        model_class,
        dataset,
        split,
        epochs,
        final_dilation=0.1,
        dilation_schedule='linear',
        score_every_n_epochs=2,
        warmup_epochs=3,
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
        
        self.final_dilation = final_dilation
        self.dilation_schedule = dilation_schedule
        self.score_every_n_epochs = score_every_n_epochs
        self.warmup_epochs = warmup_epochs
        self.sample_weights = None
        
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
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )
        
        train_metadata, _, _ = kfold_env.get_split_metadata(self.split)
        n_train_samples = len(train_metadata)
        self.sample_weights = np.ones(n_train_samples, dtype=np.float32) #Initialize all samples with weight=1.0

        
        #TRAINING LOOP
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch}/{self.epochs} ===")
            
            #Update sample weights with dilation after warmup
            if epoch >= self.warmup_epochs and epoch % self.score_every_n_epochs == 0:
                self._apply_dilation(model, kfold_env, epoch)
            
            epoch_loss = self._train_one_epoch(model, optimizer, train_gen)
            val_loss = self._validate(model, validation_gen)
            
            checkpoint_path = get_checkpoint_path(
                self.exp_name, self.model_name, self.dataset, self.split, epoch
            )
            model.save_weights(checkpoint_path)
            
            history['loss'].append(float(epoch_loss))
            history['val_loss'].append(float(val_loss))
            
            n_active = (self.sample_weights > 0).sum()
            print(f"Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Active samples: {n_active}/{n_train_samples} ({100*n_active/n_train_samples:.1f}%)")
            
        self._save_history(self.split, history)
    
    def _train_one_epoch(self, model, optimizer, train_gen):
        """Train for one epoch with per-sample weighting"""
        n_batches = len(train_gen)
        epoch_losses = []
        
        for batch_idx in range(n_batches):
            x, y = train_gen[batch_idx]
            
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_weights = self.sample_weights[start_idx:end_idx]
            batch_weights = tf.constant(batch_weights, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = model(x, training=True)
                
                per_sample_recon_loss = (
                    self._per_sample_l2_distance(x, y1) +
                    self._per_sample_l2_distance(x, y2) +
                    self._per_sample_l2_distance(x, y3) +
                    self._per_sample_l2_distance(x, y4) +
                    self._per_sample_l2_distance(x, y5)
                ) / 5.0  # Shape: (batch_size,)
                
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
                ) / 10.0  # Shape: (batch_size,)
                
                # Total per-sample loss
                per_sample_total_loss = per_sample_recon_loss + per_sample_ensemble_loss
                
                # Apply weights per-sample
                weighted_per_sample_loss = per_sample_total_loss * batch_weights
                
                # Final loss (normalize by sum of weights, not batch size)
                sum_weights = tf.reduce_sum(batch_weights)
                if sum_weights > 0:
                    loss = tf.reduce_sum(weighted_per_sample_loss) / sum_weights
                else:
                    loss = tf.constant(0.0, dtype=tf.float32)
            
            #Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_losses.append(float(loss))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{n_batches}, Loss: {loss:.4f}")
        
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
        
        # Reduce over timesteps and channels only, keep batch dimension
        distance = tf.sqrt(tf.reduce_mean(tf.square(x_demeaned - y_demeaned), axis=[1, 2]))
        return distance  # Shape: (batch_size,)
    
    def _per_sample_ensemble_distance(self, f1, f2):
        
        f1_normalized = demean(f1, axis=1)
        f2_normalized = demean(f2, axis=1)
        
        f1_normalized = l2_normalize(f1_normalized, axis=1)
        f2_normalized = l2_normalize(f2_normalized, axis=1)
        
        # Reduce over timesteps and channels, keep batch dimension
        distance = tf.sqrt(tf.reduce_mean(tf.square(f1_normalized - f2_normalized), axis=[1, 2]))
        return distance  # Shape: (batch_size,)
    
    def _apply_dilation(self, model, kfold_env, epoch):
        """Score samples and update weights"""
        print(f"\n>>> Scoring and updating weights")
        
        scores = self._score_samples(model, kfold_env, self.split)
        
        keep_ratio = self._get_keep_ratio(epoch)
        n_keep = int(keep_ratio * len(scores))
        
        sorted_indices = np.argsort(scores)[::-1]
        
        #Weights are set 1 for top N%, 0 for rest
        self.sample_weights = np.zeros(len(scores), dtype=np.float32)
        self.sample_weights[sorted_indices[:n_keep]] = 1.0
        
        train_metadata, _, _ = kfold_env.get_split_metadata(self.split)
        active_mask = self.sample_weights > 0
        n_eq_active = ((train_metadata['label'] == 'eq') & active_mask).sum()
        n_eq_total = (train_metadata['label'] == 'eq').sum()
        
        print(f"Keep ratio: {keep_ratio:.4f} ({n_keep}/{len(scores)} samples)")
        print(f"Earthquakes active: {n_eq_active}/{n_eq_total}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        if n_keep > 0:
            print(f"Score threshold: {scores[sorted_indices[n_keep-1]]:.4f}")
    
    def _score_samples(self, representation_model, kfold_env, split):
        """Score all training samples using classifier"""
        
        classifier = ClassifierMultipleAutoencoder(model=representation_model)
        
        train_gen, _, _, _ = kfold_env.get_generators(split)
        n_batches = len(train_gen)
        
        scores = []
        for batch_idx in range(n_batches):
            x, _ = train_gen[batch_idx]
            batch_scores = classifier(x, training=False)
            scores.extend(batch_scores)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Scoring: {batch_idx+1}/{n_batches}")
        
        return np.array(scores)
    
    def _get_keep_ratio(self, epoch):
        """Calculate fraction of dataset to keep active"""
        if epoch < self.warmup_epochs:
            return 1.0
        
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total_epochs = self.epochs - self.warmup_epochs
        
        progress = min(adjusted_epoch / adjusted_total_epochs, 1.0)
        
        if self.dilation_schedule == 'linear':
            keep_ratio = 1.0 - (1.0 - self.final_dilation) * progress
        else:
            keep_ratio = 1.0
        
        return keep_ratio
    
    def _save_history(self, split, history_dict):
        with open(
            get_history_csv_path(self.exp_name, self.model_name, self.dataset, split),
            "w",
        ) as f:
            hist_df = pd.DataFrame(history_dict)
            hist_df.to_csv(f)