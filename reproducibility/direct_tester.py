import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from direct_trainer import HDF5DataGenerator

class DirectTester:
    def test(self,
             representation_model_class,
             classifier_wrapper_class,
             model_weights_path: str,
             test_dataset_path: str,
             batch_size: int = 256,
             method_params: dict = {}):
        """
        Test model and return scores
        
        Returns:
            scores: Earthquake probabilities
            labels: True labels (0/1)
        """
        
        print(f"Loading model: {representation_model_class().name}")
        model = representation_model_class()
        model.compile(optimizer=tf.keras.optimizers.Adam())
        
        # Build the model before loading weights (lazy building)
        model(tf.zeros((1, 3000, 3)))
        
        model.load_weights(model_weights_path)
        print(f"Loaded weights from: {model_weights_path}")
        
        classifier = classifier_wrapper_class(model, method_params=method_params)
        
        test_gen = HDF5DataGenerator(test_dataset_path, batch_size, shuffle=False)
        
        scores = []
        labels = []
        
        print(f"\nTesting on: {test_dataset_path}")
        for i in range(len(test_gen)):
            X_batch, y_batch = test_gen[i]
            
            # Get earthquake probabilities
            batch_scores = classifier(X_batch, training=False)
            
            scores.extend(batch_scores)
            labels.extend(y_batch)
        
        return np.array(scores), np.array(labels)