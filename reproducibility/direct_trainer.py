import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple

from data_generator import BatchGenerator

from config import (
    BATCH_SIZE,
    INSTANCE_TIME_WINDOW,
    STEAD_TIME_WINDOW,
    SAMPLING_FREQ,
    FREQMIN,
    FREQMAX,
    TRAIN_VALIDATION_SPLIT,
    SUBSAMPLING_FACTOR,
    PHASE_PICK_ENSURED_CROP_RATIO,
    PHASE_ENSURING_MARGIN,
)

from directory import (
    STEAD_WAVEFORMS_HDF5_PATH,
    STEAD_METADATA_CSV_PATH,
    INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
    INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
    INSTANCE_EQ_METADATA_CSV_PATH,
    INSTANCE_NOISE_METADATA_CSV_PATH,
    PREPROCESSED_DATASET_DIRECTORY,
)


class DirectTrainer:    
    def __init__(self,
                 dataset,
                 dataset_time_window: float,
                 model_time_window: float = 30.0,
                 sampling_freq: int = SAMPLING_FREQ,
                 phase_ensuring_margin: float = PHASE_ENSURING_MARGIN,
                 phase_ensured_crop_ratio: float = PHASE_PICK_ENSURED_CROP_RATIO,
                 stead_time_window=STEAD_TIME_WINDOW,
                instance_time_window=INSTANCE_TIME_WINDOW,
                stead_waveforms_hdf5=STEAD_WAVEFORMS_HDF5_PATH,
                stead_metadata_csv=STEAD_METADATA_CSV_PATH,
                instance_eq_waveforms_hdf5=INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
                instance_no_waveforms_hdf5=INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
                instance_eq_metadata_csv=INSTANCE_EQ_METADATA_CSV_PATH,
                instance_no_metadata_csv=INSTANCE_NOISE_METADATA_CSV_PATH,):

        self.dataset = dataset
        self.dataset_time_window = dataset_time_window
        self.model_time_window = model_time_window
        self.sampling_freq = sampling_freq
        self.phase_ensuring_margin = phase_ensuring_margin
        self.phase_ensured_crop_ratio = phase_ensured_crop_ratio
        self.stead_time_window = stead_time_window
        self.instance_time_window = instance_time_window
        
    def create_subsampled_datasets(self,
                                dataset: str,
                                output_dir: str,
                                noise_percentages: List[float] = None,
                                subsampling_factor: float = 1.0,
                                maintain_constant_size: bool = True):
        """
        Create datasets with different noise percentages
        
        Args:
            dataset: Dataset name ('stead' or 'instance')
            output_dir: Output directory for preprocessed datasets
            noise_percentages: List of noise percentages (0.0 to 100.0)
                            Example: [0, 25, 50, 75, 100] for 0%, 25%, 50%, 75%, 100% noise
                            If None, creates full dataset with all available samples
            subsampling_factor: Factor to subsample the original dataset (0.0 to 1.0)
            maintain_constant_size: If True, all datasets will have the same size (LCD)
        """
        
        if dataset == "stead": 
            metadata = self._parse_stead_metadata(STEAD_METADATA_CSV_PATH)
            eq_hdf5_path = STEAD_WAVEFORMS_HDF5_PATH
            no_hdf5_path = STEAD_WAVEFORMS_HDF5_PATH
        elif dataset == "instance":
            metadata = self._parse_instance_metadata(
                INSTANCE_EQ_METADATA_CSV_PATH,
                INSTANCE_NOISE_METADATA_CSV_PATH
            )
            eq_hdf5_path = INSTANCE_EQ_WAVEFORMS_HDF5_PATH
            no_hdf5_path = INSTANCE_NOISE_WAVEFORMS_HDF5_PATH
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        eq_metadata = metadata[metadata['label'] == 'eq'].copy()
        no_metadata = metadata[metadata['label'] == 'no'].copy()
        
        print(f"Total samples before subsampling: EQ={len(eq_metadata)}, NO={len(no_metadata)}")
        
        if subsampling_factor < 1.0:
            eq_metadata = eq_metadata.sample(frac=subsampling_factor, random_state=42)
            no_metadata = no_metadata.sample(frac=subsampling_factor, random_state=42)
            print(f"After subsampling ({subsampling_factor*100:.0f}%): EQ={len(eq_metadata)}, NO={len(no_metadata)}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if noise_percentages is None or len(noise_percentages) == 0:
            print("No noise percentages specified. Creating full dataset with all available samples...")
            combined_metadata = pd.concat([eq_metadata, no_metadata])
            combined_metadata = combined_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
            combined_metadata = self._assign_crop_offsets(combined_metadata)
            
            output_file = f"FULL_DATASET_SUBSAMPLED_{int(subsampling_factor*100)}.hdf5"
            self._save_preprocessed_dataset(
                combined_metadata,
                eq_hdf5_path,
                no_hdf5_path,
                Path(output_dir) / output_file
            )
            print(f"Created full dataset: EQ={len(eq_metadata)}, NO={len(no_metadata)}, Total={len(combined_metadata)}")
            return
        
        for pct in noise_percentages:
            if not 0 <= pct <= 100:
                raise ValueError(f"Noise percentage must be between 0 and 100, got {pct}")
        
        if maintain_constant_size:
            # The LCD is the minimum of available samples
            # If we need percentages from 0-100%, we're limited by the minority class 
            # (noise for stead/instance, event for continuous data)
            lcd_size = min(len(no_metadata), len(eq_metadata))
            print(f"\nLeast common denominator (constant dataset size): {lcd_size}")
        
        # Create datasets for each noise percentage
        for noise_pct in noise_percentages:
            if maintain_constant_size:
                # Use LCD size
                total_size = min(len(no_metadata), len(eq_metadata))
                n_no = int(total_size * (noise_pct / 100.0))
                n_eq = total_size - n_no
            else:
                if noise_pct == 0:
                    # All earthquakes, no noise
                    n_no = 0
                    n_eq = len(eq_metadata)
                elif noise_pct == 100:
                    # All noise, no earthquakes
                    n_no = len(no_metadata)
                    n_eq = 0
                else:
                    # Mixed: calculate maximum possible size
                    # If we want X% noise, then (100-X)% must be earthquakes
                    # Total size is limited by: min(no_samples/(X/100), eq_samples/((100-X)/100))
                    noise_frac = noise_pct / 100.0
                    eq_frac = 1 - noise_frac
                    
                    max_from_noise = len(no_metadata) / noise_frac if noise_frac > 0 else float('inf')
                    max_from_eq = len(eq_metadata) / eq_frac if eq_frac > 0 else float('inf')
                    
                    total_size = int(min(max_from_noise, max_from_eq))
                    n_no = int(total_size * noise_frac)
                    n_eq = total_size - n_no
            
            # Ensure we don't exceed available samples
            n_no = min(n_no, len(no_metadata))
            n_eq = min(n_eq, len(eq_metadata))
            
            # Sample the data
            if n_eq > 0:
                eq_subset = eq_metadata.sample(n=n_eq, random_state=42)
            else:
                eq_subset = pd.DataFrame()
                
            if n_no > 0:
                no_subset = no_metadata.sample(n=n_no, random_state=42)
            else:
                no_subset = pd.DataFrame()
            
            # Combine and shuffle
            combined_metadata = pd.concat([eq_subset, no_subset])
            if len(combined_metadata) > 0:
                combined_metadata = combined_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
                combined_metadata = self._assign_crop_offsets(combined_metadata)
                
                output_file = f"SUBSAMPLED_{int(subsampling_factor*100)}_NOISE_{int(noise_pct)}.hdf5"
                self._save_preprocessed_dataset(
                    combined_metadata,
                    eq_hdf5_path,
                    no_hdf5_path,
                    Path(output_dir) / output_file
                )
                
                # Calculate actual percentage for verification
                actual_noise_pct = (n_no / (n_no + n_eq) * 100) if (n_no + n_eq) > 0 else 0
                print(f"Created dataset with {noise_pct}% noise (actual: {actual_noise_pct:.1f}%): "
                    f"EQ={n_eq}, NO={n_no}, Total={n_eq + n_no}")
      
    def train(self,
              model,
              train_dataset_path: str,
              val_dataset_path: str,
              epochs: int = 20,
              batch_size: int = BATCH_SIZE,
              learning_rate: float = 1e-3):
        
        train_gen = HDF5DataGenerator(train_dataset_path, batch_size, shuffle=True)
        val_gen = HDF5DataGenerator(val_dataset_path, batch_size, shuffle=False)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)
        
        Path("checkpoints").mkdir(exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/best_model.h5',
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\nStarting training...")
        print(f"Train samples: {len(train_gen) * batch_size}")
        print(f"Val samples: {len(val_gen) * batch_size}")
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history


    def _parse_stead_metadata(self, metadata_csv):
        """
        Parses the metadata of the STEAD dataset and transforms certain columns in a
        way to enable generic handling.

        Parameters
        ----------
        metadata_csv : str
            The path of the csv file for the STEAD dataset that contains the metadata of the events.

        Returns
        -------
        metadata : pandas.DataFrame
            The dataframe that contains standardized metadata of the STEAD dataset.
        """
        metadata = pd.read_csv(metadata_csv)

        metadata["source_id"] = metadata["source_id"].astype(str)
        metadata.rename({"receiver_code": "station_name"}, inplace=True)
        eq_metadata = metadata[metadata.trace_category == "earthquake_local"].copy()
        no_metadata = metadata[metadata.trace_category == "noise"].copy()

        # You need to merge eq and no samples to form chunks. Because of that you
        # need to put a column named source_id to noise waveforms. Since
        # trace_name is unique for each trace, you can use it as source_id.
        no_metadata["source_id"] = no_metadata["trace_name"]

        # Label samples.
        eq_metadata.loc[:, "label"] = "eq"
        no_metadata.loc[:, "label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])

        return standardized_metadata

    def _parse_instance_metadata(self, eq_metadata_csv, no_metadata_csv):
        """
        Parses the metadata of the instance dataset and transforms certain columns in a
        way to enable generic handling.

        Parameters
        ----------
        eq_metadata_csv : str
            The path of the csv file for the instance dataset that contains the metadata of the events.
        no_metadata_csv : str
            The path of the csv file for the instance dataset that contains the metadata of the noise.

        Returns
        -------
        metadata : pandas.DataFrame
            The dataframe that contains standardized metadata of the instance dataset.
        """
        eq_metadata = pd.read_csv(eq_metadata_csv)
        no_metadata = pd.read_csv(no_metadata_csv)

        eq_metadata["source_id"] = eq_metadata["source_id"].astype(str)
        eq_metadata.rename(
            columns={
                "station_code": "station_name",
                "trace_P_arrival_sample": "p_arrival_sample",
                "trace_S_arrival_sample": "s_arrival_sample",
            },
            inplace=True,
        )
        no_metadata.rename(columns={"station_code": "station_name"}, inplace=True)

        # Source id is not unique for noise waveforms. Instead of using
        # source id, you can use trace_name as source_id.
        no_metadata["source_id"] = no_metadata["trace_name"]

        # Label samples.
        eq_metadata["label"] = "eq"
        no_metadata["label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])
        return standardized_metadata

            
    def _assign_crop_offsets(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Exact copy of the crop offset assignment logic from KFoldEnvironment
        """
        metadata.reset_index(drop=True, inplace=True)

        # Assign crop offset hard limits
        metadata["crop_offset_low_limit"] = 0
        metadata["crop_offset_high_limit"] = self._get_ts(
            self.dataset_time_window
        ) - self._get_ts(self.model_time_window)

        # Assign crop offset min and max limits
        metadata["crop_offset_min"] = metadata["crop_offset_low_limit"]
        metadata["crop_offset_max"] = metadata["crop_offset_high_limit"]

        # Assign random crop offsets to waveforms. Since eq and no waveforms
        # need to be separately handled, we first separate them
        metadata_eq = metadata[metadata.label == "eq"]
        metadata_no = metadata[metadata.label == "no"]

        # Get number of pick ensured waveforms
        n_pick_ensured_eq_waveforms = int(
            self.phase_ensured_crop_ratio * len(metadata_eq)
        )

        # Separate eq waveforms to two groups
        metadata_eq_pick_ensured_crop = metadata_eq[0:n_pick_ensured_eq_waveforms]
        metadata_eq_random_crop = metadata_eq[n_pick_ensured_eq_waveforms:]

        # Assign max limits to crop offsets for pick ensured waveforms
        metadata_eq_pick_ensured_crop = self._assign_pick_ensured_crop_offset_ranges(
            metadata_eq_pick_ensured_crop
        )

        # Merge eq and no waveforms
        metadata = pd.concat(
            [metadata_eq_pick_ensured_crop, metadata_eq_random_crop, metadata_no],
            axis=0,
        )

        # Assign random crop offsets
        np.random.seed(0)
        metadata["crop_offset"] = metadata["crop_offset_min"] + (
            metadata["crop_offset_max"] - metadata["crop_offset_min"]
        ) * np.random.uniform(0, 1, len(metadata))

        metadata["crop_offset"] = metadata["crop_offset"].astype(int)

        # Drop auxiliary columns
        metadata.drop(
            [
                "crop_offset_low_limit",
                "crop_offset_high_limit",
                "crop_offset_min",
                "crop_offset_max",
            ],
            axis=1,
            inplace=True,
        )

        # Sort them by their idx to restore the original order
        metadata.sort_index(inplace=True)

        return metadata

    def _assign_pick_ensured_crop_offset_ranges(self, eq_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Exact copy from KFoldEnvironment - ensures P/S arrivals are in the window
        
        Args:
            eq_metadata (pd.DataFrame): Dataframe that contains the metadata of the eq waveforms.

        Returns:
            pd.DataFrame: Dataframe that contains the metadata of the eq waveforms with crop offsets
                assigned.
        """
        _eq_metadata = eq_metadata.copy()

        # Assign min limits to crop offsets for pick ensured waveforms
        _eq_metadata["crop_offset_min"] = (
            _eq_metadata[["p_arrival_sample", "s_arrival_sample"]].min(axis=1)
            + self._get_ts(self.phase_ensuring_margin)
            - self._get_ts(self.model_time_window)
        )

        _eq_metadata["crop_offset_min"] = _eq_metadata[
            ["crop_offset_min", "crop_offset_low_limit"]
        ].max(axis=1)

        # Assign max limits to crop offsets for pick ensured waveforms
        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["p_arrival_sample", "s_arrival_sample"]
        ].max(axis=1) - self._get_ts(self.phase_ensuring_margin)

        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["crop_offset_max", "crop_offset_high_limit"]
        ].min(axis=1)

        return _eq_metadata
    
    def _get_ts(self, t: float) -> int:
        """Convert time in seconds to number of timesteps"""
        return int(t * self.sampling_freq)
    
    def _save_preprocessed_dataset(self,
                                  metadata: pd.DataFrame,
                                  eq_hdf5_path: str,
                                  no_hdf5_path: str,
                                  output_path: Path):
        """Process and save dataset using existing BatchGenerator logic"""
        
        print(f"\nCreating dataset: {output_path.name}")
        print(f"Total samples: {len(metadata)} (EQ: {(metadata['label']=='eq').sum()}, "
              f"NO: {(metadata['label']=='no').sum()})")
        
        batch_gen = BatchGenerator(
            batch_size=BATCH_SIZE,
            batch_metadata=metadata,
            eq_hdf5_path=eq_hdf5_path,
            no_hdf5_path=no_hdf5_path,
            dataset_time_window=self.dataset_time_window,
            model_time_window=self.model_time_window,
            sampling_freq=self.sampling_freq,
            freqmin=FREQMIN,
            freqmax=FREQMAX,
            last_axis="channels"  # or "timesteps" depending on dataset -> change this
        )
        
        n_batches = batch_gen.num_batches()
        n_samples = n_batches * BATCH_SIZE
        
        # Create output HDF5 file
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X', (n_samples, 3000, 3), dtype='float32')
            f.create_dataset('y', (n_samples,), dtype='int32')
            
            # Save metadata as JSON
            f.create_dataset('metadata', data=metadata.to_json().encode('utf-8'))
            
            # Process batches
            sample_idx = 0
            for batch_idx in range(n_batches):
                X_batch, y_batch = batch_gen.get_batch(batch_idx)
                
                # Store in HDF5
                batch_size = X_batch.shape[0]
                f['X'][sample_idx:sample_idx+batch_size] = X_batch
                f['y'][sample_idx:sample_idx+batch_size] = y_batch
                
                sample_idx += batch_size
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx+1}/{n_batches} batches...")
        
        print(f"Dataset saved: {output_path}")


  
class HDF5DataGenerator(tf.keras.utils.Sequence):    
    def __init__(self, hdf5_path: str, batch_size: int, shuffle: bool = True):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"Dataset not found: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            self.n_samples = f['X'].shape[0]
            print(f"Loaded dataset: {hdf5_path} with {self.n_samples} samples")
        
        self.indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Sort indices to avoid HDF5 error
        sorted_indices = np.sort(batch_indices)
        
        with h5py.File(self.hdf5_path, 'r') as f:
            X = f['X'][sorted_indices]
            y = f['y'][sorted_indices]
        
        # If we need to maintain the shuffled order, we need to unsort
        if self.shuffle:
            # Get the inverse permutation to restore original order
            unsort_indices = np.empty_like(sorted_indices)
            unsort_indices[np.argsort(batch_indices)] = np.arange(len(batch_indices))
            X = X[unsort_indices]
            y = y[unsort_indices]
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)