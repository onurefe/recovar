import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from data_generator import BatchGenerator
import tempfile
import shutil

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
                                maintain_constant_size: bool = True,
                                random_state_mode: str = "fixed",
                                base_random_state: int = 42,
                                save_train_val_test_splits: bool = False,
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.2,
                                chunk_size: int = 50000): 
        """
        Create datasets with different noise percentages using chunked processing to avoid OOM
        
        Args:
            dataset: Dataset name ('stead' or 'instance')
            output_dir: Output directory for preprocessed datasets
            noise_percentages: List of noise percentages (0.0 to 100.0)
            subsampling_factor: Factor to subsample the original dataset (0.0 to 1.0)
            maintain_constant_size: If True, all datasets will have the same size (LCD)
            random_state_mode: How to handle random states for each noise percentage
            base_random_state: Base random state value
            save_train_val_test_splits: If True, saves separate train/val/test HDF5 files
            val_ratio: Validation set ratio (default: 0.2)
            test_ratio: Test set ratio (default: 0.2)
            chunk_size: Number of samples to process at once to avoid OOM (default: 50000)
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
        
        # Determine random state for initial subsampling
        if random_state_mode == "pseudorandom":
            subsample_random_state = base_random_state - 1
        else:  # fixed
            subsample_random_state = base_random_state
        
        if subsampling_factor < 1.0:
            eq_metadata = eq_metadata.sample(frac=subsampling_factor, random_state=subsample_random_state)
            no_metadata = no_metadata.sample(frac=subsampling_factor, random_state=subsample_random_state)
            print(f"After subsampling ({subsampling_factor*100:.0f}%): EQ={len(eq_metadata)}, NO={len(no_metadata)}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if noise_percentages is None or len(noise_percentages) == 0:
            print("No noise percentages specified. Creating full dataset with all available samples...")
            combined_metadata = pd.concat([eq_metadata, no_metadata])
            
            if random_state_mode == "pseudorandom":
                full_random_state = base_random_state - 2
            else:  # fixed
                full_random_state = base_random_state
                
            combined_metadata = combined_metadata.reset_index(drop=True)
            combined_metadata = self._assign_crop_offsets(combined_metadata)
            
            output_file = f"FULL_DATASET_SUBSAMPLED_{int(subsampling_factor*100)}"
            if save_train_val_test_splits:
                self._save_train_val_test_splits_chunked(  
                    combined_metadata, eq_hdf5_path, no_hdf5_path, 
                    Path(output_dir) / output_file,
                    val_ratio, test_ratio, full_random_state, chunk_size
                )
            else:
                self._save_preprocessed_dataset_chunked(  
                    combined_metadata, eq_hdf5_path, no_hdf5_path,
                    Path(output_dir) / f"{output_file}.hdf5", chunk_size
                )
            
            print(f"Created full dataset: EQ={len(eq_metadata)}, NO={len(no_metadata)}, Total={len(combined_metadata)}")
            return
        
        for pct in noise_percentages:
            if not 0 <= pct <= 100:
                raise ValueError(f"Noise percentage must be between 0 and 100, got {pct}")
        
        if maintain_constant_size:
            lcd_size = min(len(no_metadata), len(eq_metadata))
            print(f"\nLeast common denominator (constant dataset size): {lcd_size}")
        
        # Create datasets for each noise percentage
        for pct_index, noise_pct in enumerate(noise_percentages):
            
            if random_state_mode == "fixed":
                current_random_state = base_random_state
            elif random_state_mode == "pseudorandom":
                current_random_state = base_random_state + int(noise_pct)
            else:
                raise ValueError(f"Unknown random_state_mode: {random_state_mode}")
            
            if maintain_constant_size:
                total_size = min(len(no_metadata), len(eq_metadata))
                n_no = int(total_size * (noise_pct / 100.0))
                n_eq = total_size - n_no
            else:
                if noise_pct == 0:
                    n_no = 0
                    n_eq = len(eq_metadata)
                elif noise_pct == 100:
                    n_no = len(no_metadata)
                    n_eq = 0
                else:
                    noise_frac = noise_pct / 100.0
                    eq_frac = 1 - noise_frac
                    
                    max_from_noise = len(no_metadata) / noise_frac if noise_frac > 0 else float('inf')
                    max_from_eq = len(eq_metadata) / eq_frac if eq_frac > 0 else float('inf')
                    
                    total_size = int(min(max_from_noise, max_from_eq))
                    n_no = int(total_size * noise_frac)
                    n_eq = total_size - n_no
            
            n_no = min(n_no, len(no_metadata))
            n_eq = min(n_eq, len(eq_metadata))
            
            if n_eq > 0:
                eq_subset = eq_metadata.sample(n=n_eq, random_state=current_random_state)
            else:
                eq_subset = pd.DataFrame()
                
            if n_no > 0:
                no_subset = no_metadata.sample(n=n_no, random_state=current_random_state)
            else:
                no_subset = pd.DataFrame()
            
            combined_metadata = pd.concat([eq_subset, no_subset])
            if len(combined_metadata) > 0:
                combined_metadata = combined_metadata.reset_index(drop=True)
                combined_metadata = self._assign_crop_offsets(combined_metadata)
                
                output_file = f"SUBSAMPLED_{int(subsampling_factor*100)}_NOISE_{int(noise_pct)}"
                
                if save_train_val_test_splits:
                    self._save_train_val_test_splits_chunked( 
                        combined_metadata, eq_hdf5_path, no_hdf5_path,
                        Path(output_dir) / output_file,
                        val_ratio, test_ratio, current_random_state, chunk_size
                    )
                else:
                    self._save_preprocessed_dataset_chunked(
                        combined_metadata, eq_hdf5_path, no_hdf5_path,
                        Path(output_dir) / f"{output_file}.hdf5", chunk_size
                    )
                
                actual_noise_pct = (n_no / (n_no + n_eq) * 100) if (n_no + n_eq) > 0 else 0
                
                rs_info = f"random_state={current_random_state}" if current_random_state is not None else "random_state=None"
                print(f"Created dataset with {noise_pct}% noise (actual: {actual_noise_pct:.1f}%) [{rs_info}]: "
                    f"EQ={n_eq}, NO={n_no}, Total={n_eq + n_no}")

    def _save_train_val_test_splits_chunked(self, metadata: pd.DataFrame,
                                           eq_hdf5_path: str,
                                           no_hdf5_path: str,
                                           output_base_path: Path,
                                           val_ratio: float,
                                           test_ratio: float,
                                           random_state: int,
                                           chunk_size: int = 50000):
        """
        Process and save dataset as separate train/val/test HDF5 files using chunked processing.
        """
        print(f"\nCreating train/val/test splits for: {output_base_path.name}")
        print(f"Total samples: {len(metadata)} (EQ: {(metadata['label']=='eq').sum()}, "
              f"NO: {(metadata['label']=='no').sum()})")
        print(f"Using chunk size: {chunk_size}")
        
        #Determine split indices without loading all data
        labels = (metadata['label'] == 'eq').astype(int).values
        
        # Split into train/val/test
        train_val_ratio = 1.0 - test_ratio
        train_val_indices, test_indices = train_test_split(
            np.arange(len(metadata)),
            train_size=train_val_ratio,
            stratify=labels,
            random_state=random_state
        )
        
        val_ratio_adjusted = val_ratio / train_val_ratio
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio_adjusted,
            stratify=labels[train_val_indices],
            random_state=random_state
        )
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Process each split
        for split_name, split_indices in splits.items():
            split_metadata = metadata.iloc[split_indices].reset_index(drop=True)
            output_path = output_base_path.parent / f"{output_base_path.name}_{split_name}.hdf5"
            
            print(f"\nProcessing {split_name} split...")
            self._save_preprocessed_dataset_chunked(
                split_metadata, eq_hdf5_path, no_hdf5_path, output_path, chunk_size
            )
            
            # Verify the split
            with h5py.File(output_path, 'r') as f:
                n_samples = len(f['y'])
                n_eq = np.sum(f['y'][:] == 1)
                n_no = np.sum(f['y'][:] == 0)
                print(f"Saved {split_name}: {output_path} - {n_samples} samples "
                      f"(EQ: {n_eq}, NO: {n_no})")

    def _save_preprocessed_dataset_chunked(self,
                                          metadata: pd.DataFrame,
                                          eq_hdf5_path: str,
                                          no_hdf5_path: str,
                                          output_path: Path,
                                          chunk_size: int = 50000):
        """
        Process and save dataset using chunked processing
        """
        
        print(f"\nCreating dataset: {output_path.name}")
        print(f"Total samples: {len(metadata)} (EQ: {(metadata['label']=='eq').sum()}, "
              f"NO: {(metadata['label']=='no').sum()})")
        print(f"Using chunk size: {chunk_size}")
        
        # Calculate number of chunks
        n_chunks = (len(metadata) + chunk_size - 1) // chunk_size
        print(f"Will process {n_chunks} chunks")
        
        # Create output HDF5 file with estimated size
        timesteps = int(self.model_time_window * self.sampling_freq)
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets with the exact size
            f.create_dataset('X', (len(metadata), timesteps, 3), dtype='float32')
            f.create_dataset('y', (len(metadata),), dtype='int32')
            f.create_dataset('metadata', data=metadata.to_json().encode('utf-8'))
            
            sample_idx = 0
            
            # Process in chunks
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(metadata))
                chunk_metadata = metadata.iloc[start_idx:end_idx].reset_index(drop=True)
                
                print(f"Processing chunk {chunk_idx + 1}/{n_chunks} "
                      f"(samples {start_idx}:{end_idx}, size: {len(chunk_metadata)})")
                
                batch_gen = BatchGenerator(
                    batch_size=BATCH_SIZE,
                    batch_metadata=chunk_metadata,
                    eq_hdf5_path=eq_hdf5_path,
                    no_hdf5_path=no_hdf5_path,
                    dataset_time_window=self.dataset_time_window,
                    model_time_window=self.model_time_window,
                    sampling_freq=self.sampling_freq,
                    freqmin=FREQMIN,
                    freqmax=FREQMAX,
                    last_axis=self._get_last_axis()
                )
                
                n_batches = batch_gen.num_batches()
                
                # Process batches in this chunk
                for batch_idx in range(n_batches):
                    X_batch, y_batch = batch_gen.get_batch(batch_idx)
                    batch_size_actual = X_batch.shape[0]
                    
                    # Store in HDF5
                    f['X'][sample_idx:sample_idx + batch_size_actual] = X_batch
                    f['y'][sample_idx:sample_idx + batch_size_actual] = y_batch
                    
                    sample_idx += batch_size_actual
                    
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        print(f"  Processed {batch_idx}/{n_batches} batches in chunk {chunk_idx + 1}")
                
                print(f"Completed chunk {chunk_idx + 1}/{n_chunks}")
                
        print(f"Dataset saved: {output_path}")
        print(f"Total samples processed: {sample_idx}")

    def create_train_val_test_split(self, dataset_path: str, 
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_state: int = 42):
        """
        Load dataset and split into train/val/test sets with stratification.
        
        Args:
            dataset_path: Path to the HDF5 dataset
            val_ratio: Validation set ratio (default: 0.15)
            test_ratio: Test set ratio (default: 0.15)
            random_state: Random seed
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        with h5py.File(dataset_path, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
        
        # First split: train+val vs test
        train_val_ratio = 1.0 - test_ratio
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            train_size=train_val_ratio,
            stratify=y,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio_adjusted = val_ratio / train_val_ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio_adjusted,
            stratify=y_train_val,
            random_state=random_state
        )
        
        print(f"Train: {len(y_train)} samples (EQ: {np.sum(y_train==1)}, NO: {np.sum(y_train==0)})")
        print(f"Val: {len(y_val)} samples (EQ: {np.sum(y_val==1)}, NO: {np.sum(y_val==0)})")
        print(f"Test: {len(y_test)} samples (EQ: {np.sum(y_test==1)}, NO: {np.sum(y_test==0)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self,
            model,
            dataset_path: str = None,
            train_dataset_path: str = None,
            val_dataset_path: str = None,
            test_dataset_path: str = None,
            epochs: int = 20,
            batch_size: int = BATCH_SIZE,
            learning_rate: float = 1e-3,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            random_state: int = 42,
            use_hdf5_generator: bool = False):
        """
        Train the model using either in-memory or HDF5 generator approach.
        
        Args:
            model: Model to train
            dataset_path: Path to single HDF5 file (will be split into train/val/test)
            train_dataset_path: Path to train HDF5 file (use with val_dataset_path)
            val_dataset_path: Path to validation HDF5 file
            test_dataset_path: Path to test HDF5 file (optional)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_ratio: Validation ratio (used when dataset_path is provided)
            test_ratio: Test ratio (used when dataset_path is provided)
            random_state: Random state for splitting
            use_hdf5_generator: If True, use HDF5Generator to avoid loading all data into memory
        
        Returns:
            history: Training history
        """
        
        if dataset_path is not None:
            if use_hdf5_generator:
                # Create temporary split files
                temp_dir = Path("temp_splits")
                temp_dir.mkdir(exist_ok=True)
                
                # Load and split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split(
                    dataset_path=dataset_path,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    random_state=random_state
                )
                
                # Save splits to temporary files
                train_path = temp_dir / "train_temp.hdf5"
                val_path = temp_dir / "val_temp.hdf5"
                
                with h5py.File(train_path, 'w') as f:
                    f.create_dataset('X', data=X_train)
                    f.create_dataset('y', data=y_train)
                
                with h5py.File(val_path, 'w') as f:
                    f.create_dataset('X', data=X_val)
                    f.create_dataset('y', data=y_val)
                
                train_gen = HDF5Generator(str(train_path), batch_size, shuffle=False)
                val_gen = HDF5Generator(str(val_path), batch_size, shuffle=False)
                
            else:
                # In-memory approach - loads the data into memory and then splits it into val and train (Intended to be legacy now.)
                X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split(
                    dataset_path=dataset_path,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    random_state=random_state
                )

                train_gen = InMemoryDataGenerator(X_train, y_train, batch_size, shuffle=False)
                val_gen = InMemoryDataGenerator(X_val, y_val, batch_size, shuffle=False)
        
        elif train_dataset_path is not None and val_dataset_path is not None:
            if use_hdf5_generator:
                train_gen = HDF5Generator(train_dataset_path, batch_size, shuffle=False)
                val_gen = HDF5Generator(val_dataset_path, batch_size, shuffle=False)
            else:
                # Load data into memory
                with h5py.File(train_dataset_path, 'r') as f:
                    X_train = f['X'][:]
                    y_train = f['y'][:]
                
                with h5py.File(val_dataset_path, 'r') as f:
                    X_val = f['X'][:]
                    y_val = f['y'][:]
                
                train_gen = InMemoryDataGenerator(X_train, y_train, batch_size, shuffle=False)
                val_gen = InMemoryDataGenerator(X_val, y_val, batch_size, shuffle=False)
                
                print(f"Train samples: {len(y_train)}")
                print(f"Val samples: {len(y_val)}")
        
        else:
            raise ValueError("Must provide either dataset_path OR both train_dataset_path and val_dataset_path")
        
        def train_generator():
            while True:
                for i in range(len(train_gen)):
                    x, _ = train_gen[i]
                    yield x
                train_gen.on_epoch_end()

        def val_generator():
            while True:
                for i in range(len(val_gen)):
                    x, _ = val_gen[i]
                    yield x
                val_gen.on_epoch_end()

        train_x_only = train_generator()
        val_x_only = val_generator()
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)
        
        Path("checkpoints").mkdir(exist_ok=True)
        
        if dataset_path:
            
            checkpoint_name = Path(dataset_path).stem+f"_epoch_{{epoch:02d}}.h5"
        else:
            checkpoint_name = Path(train_dataset_path).stem+f"_epoch_{{epoch:02d}}.h5"
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f'checkpoints/{checkpoint_name}',
                save_weights_only=True,
                monitor='val_loss',
                verbose=1,
                save_best_only=False
            ),
        ]
        
        print("\nStarting training...")
        print(f"Train batches: {len(train_gen)}")
        print(f"Val batches: {len(val_gen)}")
        print(f"Using {'HDF5Generator' if use_hdf5_generator else 'InMemoryDataGenerator'}")
        
        history = model.fit(
            train_x_only,
            validation_data=val_x_only,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
        
        # Clean up temporary files if created
        if dataset_path and use_hdf5_generator:
            import shutil
            shutil.rmtree("temp_splits", ignore_errors=True)
        
        return history

    def _get_last_axis(self):
        """Determine the last axis based on dataset type"""
        if self.dataset == "instance":
            return "timesteps"
        else:
            return "channels"

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

class HDF5Generator(tf.keras.utils.Sequence):
    """
    Generator that loads batches from HDF5 file unlike loading entire dataset into memory in InMemoryDataGenerator.
    """
    def __init__(self, hdf5_path: str, batch_size: int, shuffle: bool = False):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        
        with h5py.File(hdf5_path, 'r') as f:
            self.n_samples = len(f['y'])
            
        self.indices = np.arange(self.n_samples)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            X_batch = f['X'][batch_indices]
            y_batch = f['y'][batch_indices]
            
        return X_batch, y_batch
    
    def on_epoch_end(self):
        pass


class InMemoryDataGenerator(tf.keras.utils.Sequence):
    """
    In-memory data generator
    """
    def __init__(self, X, y, batch_size: int, shuffle: bool = False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(y)
        self.indices = np.arange(self.n_samples)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.X[batch_indices], self.y[batch_indices]
    
    def on_epoch_end(self):
        # REMOVED SHUFFLING 
        pass