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
)
from numpy import pi as PI
import tensorflow as tf
from tensorflow import keras
from cubic_interpolation import cubic_interp1d
from detector_models import (
    AutocovarianceDetector30s,
    Ensemble5CrossCovarianceDetector30s,
)

from building_blocks import (
    normalize,
    demean,
    distance,
    latent_distance,
    AutoencoderBlock30s,
)


class AutocovarianceMonitor30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        model=None,
        monitor_params=["fc"],
        method_params={},
        name="AutocovarianceMonitor30s",
        *args,
        **kwargs
    ):
        super(AutocovarianceMonitor30s, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params

    def get_config(self):
        config = super(AutocovarianceMonitor30s, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f, y = self.model(x, training=training)

        f = demean(f, axis=1)

        monitor = {}

        if "x" in self.monitor_params:
            monitor["x"] = x

        """
        if "fcov" in self.monitor_params:
            monitor["fcov"] = tf.matmul(f, f, transpose_a=True) / (self.N_TIMESTEPS - 1)
        """
        if "f" in self.monitor_params:
            monitor["f"] = f

        if "fcov" in self.monitor_params:
            monitor["fcov"] = self.cc([f, f])

        return monitor


class AugmentationCrossCovarianceMonitor30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3
    TIME_WINDOW = 30

    def __init__(
        self,
        model=None,
        monitor_params=["fc"],
        method_params={
            "noise_augmentation_params": {"weight": 0.15, "enabled": True},
            "dispersion_augmentation_params": {
                "distance_min": 0.5e5,
                "distance_max": 0.6e5,
                "phase_velocity_min": 8e3,
                "phase_velocity_max": 10e3,
                "phase_velocity_std": 0.75e3,
                "knots": 4,
                "enabled": True,
            },
            "timewarping_params": {"std": 0.15, "knots": 4, "enabled": True},
        },
        name="AugmentationCrossCovarianceMonitor30s",
        *args,
        **kwargs
    ):
        super(AugmentationCrossCovarianceMonitor30s, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params
        self.noise_augmentation_params = method_params["noise_augmentation_params"]
        self.dispersion_augmentation_params = method_params[
            "dispersion_augmentation_params"
        ]
        self.timewarping_params = method_params["timewarping_params"]

    def get_config(self):
        config = super(AugmentationCrossCovarianceMonitor30s, self).get_config()
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
            1e-10 + energy_spectrum, axis=1, keepdims=True
        )

        mean_group_velocity = tf.reduce_sum(velocity_profile * weights, axis=1)
        delta_t = tf.expand_dims(
            (r / (1e-10 + velocity_profile)), axis=0
        ) - tf.expand_dims((r / (1e-10 + mean_group_velocity)), axis=1)

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

        xa = x
        xb = x

        if self.noise_augmentation_params["enabled"]:
            xa = self._add_noise(x)
            xb = self._add_noise(x)

        if self.dispersion_augmentation_params["enabled"]:
            xa = self._disperse(xa)
            xb = self._disperse(xb)

        if self.timewarping_params["enabled"]:
            xa = self._timewarp(xa)
            xb = self._timewarp(xb)

        fa, ya = self.model(xa, training=training)
        fb, yb = self.model(xb, training=training)

        monitor = {}

        if "xa" in self.monitor_params:
            monitor["xa"] = xa

        if "xb" in self.monitor_params:
            monitor["xb"] = xb

        if "ya" in self.monitor_params:
            monitor["ya"] = ya

        if "yb" in self.monitor_params:
            monitor["yb"] = yb

        if "fa" in self.monitor_params:
            monitor["fa"] = fa

        if "fb" in self.monitor_params:
            monitor["fb"] = fb

        if "fcov" in self.monitor_params:
            monitor["fcov"] = self.cc([fa, fb])

        if "xcov" in self.monitor_params:
            monitor["xcov"] = self.cc([xa, xb])

        if "fcorr" in self.monitor_params:
            fa_normalized = normalize(demean(fa, axis=1), axis=1)
            fb_normalized = normalize(demean(fb, axis=1), axis=1)

            monitor["fcorr"] = self.cc([fa_normalized, fb_normalized])

        return monitor


class EnsembleAugmentationCrossCovarianceMonitor30s(keras.Model):
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
            "timewarping_params": {"std": 0.1, "knots": 4, "enabled": True},
        },
        name="EnsembleAugmentationCrossCovarianceMonitor30s",
        *args,
        **kwargs
    ):
        super(EnsembleAugmentationCrossCovarianceMonitor30s, self).__init__(
            name=name, **kwargs
        )
        self.model = model
        self.monitor_params = monitor_params
        self.augmentations = method_params["augmentations"]
        self.noise_augmentation_params = method_params["noise_augmentation_params"]
        self.dispersion_augmentation_params = method_params[
            "dispersion_augmentation_params"
        ]
        self.timewarping_params = method_params["timewarping_params"]

    def get_config(self):
        config = super(EnsembleAugmentationCrossCovarianceMonitor30s, self).get_config()
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
                    fcovs.append(tf.expand_dims(fcov, axis=1))

            monitor["fcov"] = tf.concat(fcovs, axis=1)

        return monitor


class Ensemble5CrossCovarianceMonitor30s(keras.Model):
    def __init__(
        self,
        model=None,
        monitor_params=[],
        name="Ensemble5CrossCovarianceMonitor30s",
        *args,
        **kwargs
    ):
        super(Ensemble5CrossCovarianceMonitor30s, self).__init__(name=name, **kwargs)
        self.model = model
        self.monitor_params = monitor_params

    def get_config(self):
        config = super(Ensemble5CrossCovarianceMonitor30s, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.cc = CrossCovarianceCircular()

    def call(self, inputs, training=False):
        x = self.model.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f1, f2, f3, f4, f5, y1, y2, y3, y4, y5 = self.model(x, training=training)

        monitor = {}

        if "x" in self.monitor_params:
            monitor["x"] = x

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
