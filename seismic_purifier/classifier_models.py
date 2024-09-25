import tensorflow as tf
from tensorflow import keras
import numpy as np
from seismic_purifier.config import BATCH_SIZE, SAMPLING_FREQ
from seismic_purifier.layers import CrossCovarianceCircular
from seismic_purifier.cubic_interpolation import cubic_interp1d
from seismic_purifier.utils import l2_normalize
from itertools import combinations

def gaussian_window(timesteps, sigma=1.25, axis=1):
    t = np.expand_dims(
        np.linspace(-timesteps/2., timesteps/2., timesteps), axis=0
    )
    g = np.exp(-np.power(t, 2.0) / (2 * np.power(sigma, 2.0)))
    return g / np.sum(g, axis=axis, keepdims=True)
    
def eq_metric(fcov):
    n_timesteps = np.shape(fcov)[1]
    g = gaussian_window(n_timesteps)
    z = np.maximum(np.mean(fcov * g, axis=1), 0)
    return (1. - np.exp(-z))

class ClassifierAutocovariance(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        model=None,
        method_params={},
        name="autocovariance",
        *args,
        **kwargs
    ):
        super(ClassifierAutocovariance, self).__init__(name=name, **kwargs)
        self.model = model

    def get_config(self):
        config = super(ClassifierAutocovariance, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f, __ = self.model(x, training=training)
        fcov = self.cc([f, f])

        return eq_metric(fcov)

class ClassifierAugmentedAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3
    TIME_WINDOW = 30

    def __init__(
        self,
        model=None,
        method_params={"augmentations": 5, "std": 0.15, "knots": 4},
        name="classifier_augmentation_cross_covariances",
        *args,
        **kwargs
    ):
        super(ClassifierAugmentedAutoencoder, self).__init__(name=name, **kwargs)
        self.model = model
        self.method_params = method_params

    def get_config(self):
        config = super(ClassifierAugmentedAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.cc = CrossCovarianceCircular()

        self.t = tf.linspace(
            0.0,
            self.TIME_WINDOW * ((self.N_TIMESTEPS - 1.0) / self.N_TIMESTEPS),
            self.N_TIMESTEPS,
        )
        self.t_knots = tf.linspace(
            0.0,
            self.TIME_WINDOW * ((self.N_TIMESTEPS - 1.0) / self.N_TIMESTEPS),
            self.method_params["knots"] + 2,
        )

    def _timewarp(self, x):
        knot_values = tf.random.normal(
            [self.method_params["knots"]],
            stddev=self.method_params["std"],
        )

        knot_values = tf.concat([tf.zeros([1]), knot_values, tf.zeros([1])], axis=0)

        t_warped = self.t + cubic_interp1d(self.t, self.t_knots, knot_values)
        t_warped = t_warped - tf.reduce_min(t_warped, keepdims=True)
        t_warped = (
            (self.TIME_WINDOW - (2.0 / SAMPLING_FREQ))
            / tf.reduce_max(t_warped, keepdims=True)
        ) * t_warped

        idx_floor = tf.math.floor(t_warped * SAMPLING_FREQ)
        interp_point_distance_to_floor = (t_warped * SAMPLING_FREQ) - idx_floor

        weight_floor = 1.0 / (1e-37 + interp_point_distance_to_floor)
        weight_ceil = 1.0 / (1e-37 + (1.0 - interp_point_distance_to_floor))

        weight_floor = weight_floor / (weight_floor + weight_ceil)
        weight_floor = tf.expand_dims(tf.expand_dims(weight_floor, axis=0), axis=2)

        x = weight_floor * tf.gather(x, tf.cast(idx_floor, tf.int32), axis=1) + (
            1.0 - weight_floor
        ) * tf.gather(x, (1 + tf.cast(idx_floor, tf.int32)), axis=1)

        return x

    def _cross_covariance_ensemble_mean(self, f_list):
        covariances = []
        pairs = combinations(list(range(len(f_list))), 2)

        for pair in pairs:
            covariance = self.cc([f_list[pair[0]], f_list[pair[1]]])
            covariances.append(covariance)

        return tf.reduce_mean(tf.convert_to_tensor(covariances), axis=0)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = l2_normalize(x)

        augmented_x = [0] * self.method_params["augmentations"]
        augmented_f = [0] * self.method_params["augmentations"]
        augmented_y = [0] * self.method_params["augmentations"]

        for i in range(self.method_params["augmentations"]):
            augmented_x[i] = self._timewarp(x)

            augmented_f[i], augmented_y[i] = self.model(
                augmented_x[i], training=training
            )

        fcov = self._cross_covariance_ensemble_mean(augmented_f)
        return eq_metric(fcov)

class ClassifierMultipleAutoencoder(keras.Model):
    def __init__(
        self,
        model=None,
        method_params={},
        name="representation_cross_covariances",
        *args,
        **kwargs
    ):
        super(ClassifierMultipleAutoencoder, self).__init__(name=name, **kwargs)
        self.model = model

    def get_config(self):
        config = super(ClassifierMultipleAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):
        self.cc = CrossCovarianceCircular()

    def _cross_covariance_ensemble_mean(self, f_list):
        covariances = []
        pairs = combinations(list(range(len(f_list))), 2)

        for pair in pairs:
            covariance = self.cc([f_list[pair[0]], f_list[pair[1]]])
            covariances.append(covariance)

        return tf.reduce_mean(tf.convert_to_tensor(covariances), axis=0)

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f1, f2, f3, f4, f5, __, __, __, __, __ = self.model(x, training=training)
        fcov = self._cross_covariance_ensemble_mean([f1, f2, f3, f4, f5])
        
        return eq_metric(fcov)