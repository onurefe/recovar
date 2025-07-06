import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from direct_trainer import HDF5Generator, InMemoryDataGenerator
from evaluator import CropOffsetFilter
import pandas as pd

class DirectTester:
    def test(self,
             representation_model_class,
             classifier_wrapper_class,
             model_weights_path: str,
             test_dataset_path: str,
             batch_size: int = 256,
             method_params: dict = {},
             use_hdf5_generator: bool = True,
             input_shape: tuple = (3000, 3),
             apply_crop_offset_filter: bool = True):
        """
        Test model and return scores with optional crop offset filtering
        
        Args:
            representation_model_class: The representation learning model class
            classifier_wrapper_class: The classifier wrapper class  
            model_weights_path: Path to saved model weights
            test_dataset_path: Path to test dataset HDF5 file
            batch_size: Batch size for testing
            method_params: Additional parameters for the classifier
            use_hdf5_generator: Whether to use HDF5Generator (recommended for large datasets)
            input_shape: Expected input shape (timesteps, channels)
            apply_crop_offset_filter: Whether to apply crop offset filter to ensure P arrivals are in window
        
        Returns:
            scores: Earthquake probabilities
            labels: True labels (0/1)
            filtered_indices: Indices that passed the filter (None if no filter applied)
        """
        
        print(f"Loading model: {representation_model_class().name}")
        model = representation_model_class()
        model.compile(optimizer=tf.keras.optimizers.Adam())
        
        # Build the model before loading weights (lazy building)
        dummy_input = tf.zeros((1,) + input_shape)
        model(dummy_input)
        
        if Path(model_weights_path).exists():
            model.load_weights(model_weights_path)
            print(f"Loaded weights from: {model_weights_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")
        
        classifier = classifier_wrapper_class(model, method_params=method_params)
        
        metadata = None
        filtered_indices = None
        
        if apply_crop_offset_filter:
            print("Loading metadata for crop offset filtering...")
            with h5py.File(test_dataset_path, 'r') as f:
                if 'metadata' in f:
                    metadata_json = f['metadata'][()].decode('utf-8')
                    metadata = pd.read_json(metadata_json)
                    
                    crop_filter = CropOffsetFilter()
                    filtered_metadata = crop_filter.apply(metadata)
                    
                    #Get indices that passed the filter
                    filtered_indices = filtered_metadata.index.values
                    
                    print(f"Crop offset filter: {len(filtered_indices)}/{len(metadata)} samples passed")
                else:
                    print("Warning: No metadata found in HDF5 file. Skipping crop offset filter.")
                    apply_crop_offset_filter = False
        
        if use_hdf5_generator:
            test_gen = HDF5Generator(test_dataset_path, batch_size, shuffle=False)
        else:
            # Load data into memory
            with h5py.File(test_dataset_path, 'r') as f:
                X_test = f['X'][:]
                y_test = f['y'][:]
            test_gen = InMemoryDataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        scores = []
        labels = []
        
        print(f"\nTesting on: {test_dataset_path}")
        print(f"Number of batches: {len(test_gen)}")
        print(f"Using {'HDF5Generator' if use_hdf5_generator else 'InMemoryDataGenerator'}")
        
        # Process in batches
        for i in range(len(test_gen)):
            X_batch, y_batch = test_gen[i]
            
            batch_scores = classifier(X_batch, training=False)
            
            if hasattr(batch_scores, 'numpy'):
                batch_scores = batch_scores.numpy()
            
            # Ensure scores are 1D
            if len(batch_scores.shape) > 1:
                batch_scores = batch_scores.flatten()
            
            scores.extend(batch_scores)
            labels.extend(y_batch)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(test_gen):
                print(f"Processed {i+1}/{len(test_gen)} batches...")
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        if apply_crop_offset_filter and filtered_indices is not None:
            scores = scores[filtered_indices]
            labels = labels[filtered_indices]
            
            print(f"\nAfter crop offset filtering:")
            print(f"Total samples: {len(labels)}")
            print(f"Earthquake samples: {np.sum(labels == 1)}")
            print(f"Noise samples: {np.sum(labels == 0)}")
        else:
            print(f"\nTesting completed:")
            print(f"Total samples: {len(labels)}")
            print(f"Earthquake samples: {np.sum(labels == 1)}")
            print(f"Noise samples: {np.sum(labels == 0)}")
        
        print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        
        return scores, labels