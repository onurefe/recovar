from config import (
    BATCH_SIZE,
    FREQMIN,
    FREQMAX,
    SAMPLING_FREQ,
)
from layers import CrossCovarianceCircular
from numpy import pi as PI
import tensorflow as tf
from tensorflow import keras
from cubic_interpolation import cubic_interp1d
from utils import l2_normalize
from itertools import combinations


class StaLta(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        monitored_params=["max_sta_over_lta"],
        method_params={
            "short_term_window": 1.0,
            "long_term_window": 4.0,
            "sampling_frequency": 100.0,
        },
        eps=1e-27,
        name="StaLta",
        *args,
        **kwargs
    ):
        super(StaLta, self).__init__(name=name, **kwargs)
        self.monitored_params = monitored_params
        self.method_params = method_params
        self.eps = eps

    def get_config(self):
        config = super(StaLta, self).get_config()
        return config

    def build(self, input_shape=None):
        pass

    def call(self, inputs, training=False):
        x = tf.cast(inputs, dtype=tf.float32)

        sta_over_lta = self._sta_lta(x)

        monitor = {}

        if "max_sta_over_lta" in self.monitored_params:
            monitor["max_sta_over_lta"] = tf.reduce_max(sta_over_lta, axis=1)

        return monitor

    def _average(self, x, window_samples):
        cumulative_power = tf.cumsum(x, axis=(1))
        sum_in_window = (
            cumulative_power[:, window_samples:] - cumulative_power[:, :-window_samples]
        )

        sum_in_window = tf.concat(
            [
                cumulative_power[:, :window_samples],
                sum_in_window,
            ],
            axis=1,
        )

        return sum_in_window / window_samples

    def _clear_elements(self, x, start_timestep, stop_timestep):
        n_timesteps = tf.shape(x)[1]

        mask = tf.range(start=0, limit=n_timesteps)
        mask = tf.cast((mask < start_timestep) | (mask >= stop_timestep), tf.int32)

        return x * mask[tf.newaxis, :]

    def _sta_lta(self, x):
        nsta = self._get_discrete_time(self.method_params["short_term_window"])
        nlta = self._get_discrete_time(self.method_params["long_term_window"])

        sta = self._average(tf.reduce_sum(x**2, axis=2), nsta)
        lta = self._average(tf.reduce_sum(x**2, axis=2), nlta)

        sta_over_lta = sta / (self.eps + lta)
        sta_over_lta = self._clear_elements(sta_over_lta, 0, nlta)

        return sta_over_lta

    def _get_discrete_time(self, time_window):
        return int(time_window * self.method_params["sampling_freq"])


class Autocovariance(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        model=None,
        monitored_params=["fcov"],
        method_params={},
        name="autocovariance",
        *args,
        **kwargs
    ):
        super(Autocovariance, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitored_params = monitored_params

    def get_config(self):
        config = super(Autocovariance, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f, y = self.model(x, training=training)

        monitor = {}

        if "xcov" in self.monitored_params:
            monitor["xcov"] = self.cc([tf.abs(x), tf.abs(x)])

        if "x" in self.monitored_params:
            monitor["x"] = x

        if "f" in self.monitored_params:
            monitor["f"] = f

        if "fcov" in self.monitored_params:
            monitor["fcov"] = self.cc([f, f])

        if "y" in self.monitored_params:
            monitor["y"] = y

        if "ycov" in self.monitored_params:
            monitor["ycov"] = self.cc([y, y])

        return monitor


class AugmentationCrossCovariances(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3
    TIME_WINDOW = 30

    def __init__(
        self,
        model=None,
        monitored_params=["fcov"],
        method_params={"augmentations": 5, "std": 0.15, "knots": 4},
        name="augmentation_cross_covariances",
        *args,
        **kwargs
    ):
        super(AugmentationCrossCovariances, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitored_params = monitored_params
        self.method_params = method_params

    def get_config(self):
        config = super(AugmentationCrossCovariances, self).get_config()
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

        monitor = {}

        if "aug_x" in self.monitored_params:
            monitor["aug_x"] = tf.convert_to_tensor(augmented_x)

        if "aug_f" in self.monitored_params:
            monitor["aug_f"] = tf.convert_to_tensor(augmented_f)

        if "aug_y" in self.monitored_params:
            monitor["aug_y"] = tf.convert_to_tensor(augmented_y)

        if "fcov" in self.monitored_params:
            monitor["fcov"] = self._cross_covariance_ensemble_mean(augmented_f)

        return monitor


class RepresentationCrossCovariances(keras.Model):
    def __init__(
        self,
        model=None,
        monitored_params=[],
        method_params={},
        name="representation_cross_covariances",
        *args,
        **kwargs
    ):
        super(RepresentationCrossCovariances, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitored_params = monitored_params

    def get_config(self):
        config = super(RepresentationCrossCovariances, self).get_config()
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

        f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = self.model(x, training=training)

        monitor = {}

        if "x" in self.monitored_params:
            monitor["x"] = x

        if "y1" in self.monitored_params:
            monitor["y1"] = y1

        if "y2" in self.monitored_params:
            monitor["y2"] = y2

        if "y3" in self.monitored_params:
            monitor["y3"] = y3

        if "y4" in self.monitored_params:
            monitor["y4"] = y4

        if "y5" in self.monitored_params:
            monitor["y5"] = y5

        if "f1" in self.monitored_params:
            monitor["f1"] = f1

        if "f2" in self.monitored_params:
            monitor["f2"] = f2

        if "f3" in self.monitored_params:
            monitor["f3"] = f3

        if "f4" in self.monitored_params:
            monitor["f4"] = f4

        if "f5" in self.monitored_params:
            monitor["f5"] = f5

        if "fcov" in self.monitored_params:
            monitor["fcov"] = self._cross_covariance_ensemble_mean([f1, f2, f3, f4, f5])

        return monitor
