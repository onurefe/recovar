from config import (
    BATCH_SIZE,
    FREQMIN,
    FREQMAX,
    SAMPLING_FREQ,
)
from layers import (
    Downsample,
    ResIdentity,
    Conv,
    Upsample,
    Padding,
    CrossCovarianceCircular,
    Dispersion,
)
from numpy import pi as PI
import tensorflow as tf
from tensorflow import keras
from cubic_interpolation import cubic_interp1d
from training_models import (
    Autoencoder,
    DenoisingAutoencoder,
    RepresentationEnsemble,
)

from building_blocks import (
    normalize,
    demean,
    distance,
    latent_distance,
    AutoencoderBlock,
)


@tf.keras.utils.register_keras_serializable()
class StaLtaDetector(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        short_term_window=1.0,
        long_term_window=4.0,
        sampling_freq=100.0,
        name="stalta_detector",
        *args,
        **kwargs
    ):
        super(StaLtaDetector, self).__init__(name=name, **kwargs)
        self._short_term_window = short_term_window
        self._long_term_window = long_term_window
        self.sampling_freq = sampling_freq

    def get_config(self):
        config = super(StaLtaDetector, self).get_config()
        config.update(
            {
                "short_term_window": self._short_term_window,
                "long_term_window": self._long_term_window,
                "sampling_freq": self.sampling_freq,
            }
        )
        return config

    def build(self, input_shape=None):
        pass

    def call(self, input):
        return self._sta_lta(
            input,
            int(self._short_term_window * self.sampling_freq),
            int(self._long_term_window * self.sampling_freq),
        )

    @property
    def short_term_window(self):
        return self._short_term_window

    @short_term_window.setter
    def short_term_window(self, value):
        self._short_term_window = value

    @property
    def long_term_window(self):
        return self._long_term_window

    @long_term_window.setter
    def long_term_window(self, value):
        self._long_term_window = value

    def _sta_lta(self, x, nsta, nlta):
        """
        Computes the standard STA/LTA from a given input array a. The length of
        the STA is given by nsta in samples, respectively is the length of the
        LTA given by nlta in samples. Written in Python.

        .. note::

            There exists a faster version of this trigger wrapped in C
            called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!

        :type a: NumPy :class:`~numpy.ndarray`
        :param a: Seismic Trace
        :type nsta: int
        :param nsta: Length of short time average window in samples
        :type nlta: int
        :param nlta: Length of long time average window in samples
        :rtype: NumPy :class:`~numpy.ndarray`
        :return: Characteristic function of classic STA/LTA
        """
        # The cumulative sum can be exploited to calculate a moving average (the
        # cumsum function is quite efficient)
        sta = tf.cumsum(tf.reduce_sum(x**2, axis=2), axis=(1))

        # Copy for LTA
        lta = tf.identity(sta)

        # Compute the STA and the LTA
        sta = tf.concat([sta[:, :nsta], (sta[:, nsta:] - sta[:, :-nsta])], axis=1)
        sta /= nsta
        lta = tf.concat([lta[:, :nlta], (lta[:, nlta:] - lta[:, :-nlta])], axis=1)
        lta /= nlta

        # Pad zeros
        batch_size = tf.shape(sta)[0]
        sta = tf.concat(
            [tf.zeros((batch_size, nlta), dtype=tf.float32), sta[:, nlta:]],
            axis=1,
        )

        return sta / (1e-20 + lta)


class StaLtaMonitor(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        model=None,
        monitor_params=["score"],
        method_params={},
        name="StaLtaMonitor",
        *args,
        **kwargs
    ):
        super(StaLtaMonitor, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params
        self.method_params = method_params

    def get_config(self):
        config = super(StaLtaMonitor, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        if "short_term_window" in self.method_params:
            self.model.short_term_window = self.method_params["short_term_window"]

        if "long_term_window" in self.method_params:
            self.model.long_term_window = self.method_params["long_term_window"]

    def call(self, inputs, training=False):
        x = tf.cast(inputs, dtype=tf.float32)

        y = self.model(x, training=training)

        monitor = {}

        if "score" in self.monitor_params:
            monitor["score"] = tf.reduce_max(y, axis=1)

        if "y" in self.monitor_params:
            monitor["y"] = y

        return monitor


class Autocovariance(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        model=None,
        monitor_params=["fc"],
        method_params={},
        name="Autocovariance",
        *args,
        **kwargs
    ):
        super(Autocovariance, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params

    def get_config(self):
        config = super(Autocovariance, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def add_eps_noise(self, x):
        x_noised = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=1e-10)
        return x_noised

    def normalize_std(self, x):
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        x = x / std
        return x

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize_std(self.add_eps_noise(x))

        f, y = self.model(x, training=training)

        f = demean(f, axis=1)

        monitor = {}

        if "xcov" in self.monitor_params:
            monitor["xcov"] = self.cc([tf.abs(x), tf.abs(x)])

        if "x" in self.monitor_params:
            monitor["x"] = x

        if "f" in self.monitor_params:
            monitor["f"] = f

        if "fcov" in self.monitor_params:
            monitor["fcov"] = self.cc([f, f])

        if "y" in self.monitor_params:
            monitor["y"] = y

        if "ycov" in self.monitor_params:
            monitor["ycov"] = self.cc([y, y])

        return monitor


class AugmentationEnsembleCrossCovariances(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3
    TIME_WINDOW = 30

    def __init__(
        self,
        model=None,
        monitor_params=["fc"],
        method_params={
            "augmentations": 5,
            "noise_augmentation_params": {"weight": 0.1, "enabled": True},
            "dispersion_augmentation_params": {
                "distance_min": 0.5e5,
                "distance_max": 0.6e5,
                "phase_velocity_min": 8e3,
                "phase_velocity_max": 10e3,
                "phase_velocity_std": 1e3,
                "knots": 4,
                "enabled": True,
            },
            "timewarping_params": {"std": 0.15, "knots": 4, "enabled": True},
        },
        name="augmentation_ensemble_cross_covariances",
        *args,
        **kwargs
    ):
        super(AugmentationEnsembleCrossCovariances, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params
        self.augmentations = method_params["augmentations"]
        self.noise_augmentation_params = method_params["noise_augmentation_params"]
        self.dispersion_augmentation_params = method_params[
            "dispersion_augmentation_params"
        ]
        self.timewarping_params = method_params["timewarping_params"]

    def get_config(self):
        config = super(AugmentationEnsembleCrossCovariances, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.cc = CrossCovarianceCircular()

        self.logf_knots = tf.linspace(
            tf.math.log(FREQMIN),
            tf.math.log(FREQMAX),
            self.dispersion_augmentation_params["knots"],
        )

        self.wave_velocity_knots_ideal = self._linear_interpolate(
            self.logf_knots,
            tf.math.log(FREQMIN),
            tf.math.log(FREQMAX),
            self.dispersion_augmentation_params["phase_velocity_min"],
            self.dispersion_augmentation_params["phase_velocity_max"],
        )

        self.f = self._fftf()

        self.t = tf.linspace(
            0.0,
            self.TIME_WINDOW * ((self.N_TIMESTEPS - 1.0) / self.N_TIMESTEPS),
            self.N_TIMESTEPS,
        )
        self.t_knots = tf.linspace(
            0.0,
            self.TIME_WINDOW * ((self.N_TIMESTEPS - 1.0) / self.N_TIMESTEPS),
            self.timewarping_params["knots"] + 2,
        )

    def _fftf(self):
        """
        Similar to numpy fftfreq function but returning w.
        """
        n = self.N_TIMESTEPS
        d = 1.0 / SAMPLING_FREQ
        f1 = tf.range(0, (n + 1) // 2, delta=1.0) / (tf.cast(n, tf.float32) * d)
        f2 = tf.range(-(n - 1) // 2, 0, delta=1.0) / (tf.cast(n, tf.float32) * d)
        f = tf.concat([f1, f2], axis=0)
        return f

    @staticmethod
    def _linear_interpolate(x, x1, x2, y1, y2):
        return y1 + (x - x1) * ((y2 - y1) / (x2 - x1))

    def _get_velocity_profile(self):
        wave_velocity_knots = self.wave_velocity_knots_ideal + tf.random.normal(
            mean=0.0,
            stddev=self.dispersion_augmentation_params["phase_velocity_std"],
            shape=[self.dispersion_augmentation_params["knots"]],
        )

        logf = tf.math.log(tf.maximum(tf.minimum(tf.abs(self.f), FREQMAX), FREQMIN))

        wave_velocity_profile = cubic_interp1d(
            logf, self.logf_knots, wave_velocity_knots
        )

        return wave_velocity_profile

    def _timewarp(self, x):
        knot_values = tf.random.normal(
            [self.timewarping_params["knots"]],
            stddev=self.timewarping_params["std"],
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

    def _disperse(self, x):
        r = tf.random.uniform(
            [],
            minval=self.dispersion_augmentation_params["distance_min"],
            maxval=self.dispersion_augmentation_params["distance_max"],
        )

        velocity_profile = self._get_velocity_profile()

        x = tf.transpose(x, perm=[0, 2, 1])
        xw = tf.signal.fft(tf.cast(x, tf.complex128))

        energy_spectrum = tf.cast(tf.square(tf.math.abs(xw)), tf.float32)
        energy_spectrum = tf.reduce_sum(energy_spectrum, axis=1)

        weights = energy_spectrum / tf.reduce_sum(
            energy_spectrum, axis=1, keepdims=True
        )

        mean_group_velocity = tf.reduce_sum(velocity_profile * weights, axis=1)
        delta_t = tf.expand_dims((r / velocity_profile), axis=0) - tf.expand_dims(
            (r / mean_group_velocity), axis=1
        )

        phase_shifts = -2.0 * PI * self.f * delta_t
        phase_shifts = tf.expand_dims(phase_shifts, axis=1)

        x = tf.signal.ifft(xw * tf.math.exp(1j * tf.cast(phase_shifts, tf.complex128)))
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.cast(x, tf.float32)

        return x

    def _add_noise(self, x):
        n = tf.random.normal(
            tf.shape(x),
            mean=0.0,
            stddev=1,
            dtype=tf.float32,
        )

        n = normalize(normalize(n, axis=1), axis=[1, 2])

        x = x + n * self.noise_augmentation_params["weight"]
        x = normalize(normalize(x, axis=1), axis=[1, 2])
        return x

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=1e-10)
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        x = x / std

        augmented_x = [0] * self.augmentations
        augmented_f = [0] * self.augmentations
        augmented_y = [0] * self.augmentations

        for i in range(self.augmentations):
            _x = x

            if self.noise_augmentation_params["enabled"]:
                _x = self._add_noise(_x)

            if self.dispersion_augmentation_params["enabled"]:
                _x = self._disperse(_x)

            if self.timewarping_params["enabled"]:
                _x = self._timewarp(_x)

            augmented_x[i] = _x

            augmented_f[i], augmented_y[i] = self.model(_x, training=training)

        monitor = {}

        if "aug_x" in self.monitor_params:
            monitor["aug_x"] = tf.convert_to_tensor(augmented_x)

        if "aug_f" in self.monitor_params:
            monitor["aug_f"] = tf.convert_to_tensor(augmented_f)

        if "aug_y" in self.monitor_params:
            monitor["aug_y"] = tf.convert_to_tensor(augmented_y)

        fcovs = []
        if "fcov" in self.monitor_params:
            for i in range(self.augmentations):
                for j in range(self.augmentations):
                    if j >= i:
                        continue

                    fcov = self.cc([augmented_f[i], augmented_f[j]])
                    fcovs.append(fcov)

            monitor["fcov"] = tf.reduce_mean(tf.convert_to_tensor(fcovs), axis=0)

        return monitor


class RepresentationEnsembleCrossCovariances(keras.Model):
    def __init__(
        self,
        model=None,
        monitor_params=[],
        method_params={},
        name="representation_ensemble_cross_covariances",
        *args,
        **kwargs
    ):
        super(RepresentationEnsembleCrossCovariances, self).__init__(
            name=name, **kwargs
        )
        self.model = model
        self.monitor_params = monitor_params

    def get_config(self):
        config = super(RepresentationEnsembleCrossCovariances, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x_noised = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=1e-10)
        std = tf.math.reduce_std(x_noised, axis=1, keepdims=True)

        x_noised = x_noised / std

        f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = self.model(x_noised, training=training)

        monitor = {}

        if "x" in self.monitor_params:
            monitor["x"] = x_noised

        if "y1" in self.monitor_params:
            monitor["y1"] = y1

        if "y2" in self.monitor_params:
            monitor["y2"] = y2

        if "y3" in self.monitor_params:
            monitor["y3"] = y3

        if "y4" in self.monitor_params:
            monitor["y4"] = y4

        if "y5" in self.monitor_params:
            monitor["y5"] = y5

        if "f1" in self.monitor_params:
            monitor["f1"] = f1

        if "f2" in self.monitor_params:
            monitor["f2"] = f2

        if "f3" in self.monitor_params:
            monitor["f3"] = f3

        if "f4" in self.monitor_params:
            monitor["f4"] = f4

        if "f5" in self.monitor_params:
            monitor["f5"] = f5

        if "fcov" in self.monitor_params:
            monitor["fcov"] = (
                self.cc([f1, f2])
                + self.cc([f1, f3])
                + self.cc([f2, f3])
                + self.cc([f1, f4])
                + self.cc([f2, f4])
                + self.cc([f3, f4])
                + self.cc([f1, f5])
                + self.cc([f2, f5])
                + self.cc([f3, f5])
                + self.cc([f4, f5])
            ) / 10.0

        return monitor
