import tensorflow as tf
from math import pi
import numpy as np
from cubic_interpolation import cubic_interp1d


@tf.keras.utils.register_keras_serializable()
class Padding(tf.keras.layers.Layer):
    def __init__(self, size=[0, 0], type="REFLECT", name="padding", *args, **kwargs):
        super(Padding, self).__init__(name=name, **kwargs)
        self.size = size
        self.type = type

    def call(self, x):
        return tf.pad(
            x,
            [
                [0, 0],
                self.size,
                [0, 0],
            ],
            mode="REFLECT",
        )


@tf.keras.utils.register_keras_serializable()
class Conv(tf.keras.layers.Layer):
    def __init__(
        self, num_of_filters, filter_kernel_size, name="conv", *args, **kwargs
    ):
        super(Conv, self).__init__(name=name, **kwargs)
        self.padding = Padding([filter_kernel_size // 2, filter_kernel_size // 2])
        self.conv = tf.keras.layers.Conv1D(
            num_of_filters, filter_kernel_size, padding="valid"
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, input_tensor, training=False):
        x = self.padding(input_tensor)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


@tf.keras.utils.register_keras_serializable()
class Upsample(tf.keras.layers.Layer):
    def __init__(
        self, num_of_filters, filter_kernel_size, name="upsample", *args, **kwargs
    ):
        super(Upsample, self).__init__(name=name, **kwargs)
        self.up = tf.keras.layers.UpSampling1D(size=2)
        self.padding = Padding([filter_kernel_size // 2, filter_kernel_size // 2])
        self.conv = tf.keras.layers.Conv1D(
            num_of_filters, filter_kernel_size, padding="valid"
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, input_tensor, training=False):
        x = self.up(input_tensor)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


@tf.keras.utils.register_keras_serializable()
class UpsampleNoactivation(tf.keras.layers.Layer):
    def __init__(
        self,
        num_of_filters,
        filter_kernel_size,
        name="upsample_noactivation",
        *args,
        **kwargs
    ):
        super(UpsampleNoactivation, self).__init__(name=name, **kwargs)
        self.up = tf.keras.layers.UpSampling1D(size=2)
        self.padding = Padding([filter_kernel_size // 2, filter_kernel_size // 2])
        self.conv = tf.keras.layers.Conv1D(
            num_of_filters, filter_kernel_size, padding="valid"
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, input_tensor, training=False):
        x = self.up(input_tensor)
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x, training=training)

        return x


@tf.keras.utils.register_keras_serializable()
class Downsample(tf.keras.layers.Layer):
    def __init__(
        self, num_of_filters, filter_kernel_size, name="downsample", *args, **kwargs
    ):
        super(Downsample, self).__init__(name=name, **kwargs)
        self.padding = Padding([filter_kernel_size // 2, filter_kernel_size // 2])
        self.conv = tf.keras.layers.Conv1D(
            num_of_filters,
            filter_kernel_size,
            strides=2,
            padding="valid",
        )
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, x, training=False):
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


@tf.keras.utils.register_keras_serializable()
class ResIdentity(tf.keras.layers.Layer):
    def __init__(
        self, num_of_filters, filter_kernel_size, name="res_identity", *args, **kwargs
    ):
        super(ResIdentity, self).__init__(name=name, **kwargs)
        self.padding = Padding([filter_kernel_size // 2, filter_kernel_size // 2])
        self.conv1 = tf.keras.layers.Conv1D(
            num_of_filters,
            filter_kernel_size,
            padding="valid",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2 = tf.keras.layers.Conv1D(
            num_of_filters, filter_kernel_size, padding="valid"
        )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.filter_kernel_size = filter_kernel_size

    def call(self, x, training=False):
        # Skip value.
        x_skip = x

        # Layer 1
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # Layer 2
        x = self.padding(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        # Add Residue
        x = x + x_skip

        # Final nonlinearity
        x = tf.nn.relu(x)

        return x


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, n_timesteps, n_features):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.pos_encoding = self._positional_encoding(
            length=n_timesteps, depth=n_features
        )

    def call(self, x):
        # Set relative scale and positional encoding.
        x *= tf.math.sqrt(tf.cast(self.n_features, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, : self.n_timesteps, :]
        return x

    @staticmethod
    def _positional_encoding(length, depth):
        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class Dispersion(tf.keras.layers.Layer):
    def __init__(
        self,
        distance_min=0.5e5,
        distance_max=0.6e5,
        phase_velocity_min=8e3,
        phase_velocity_max=10e3,
        phase_velocity_std=0.75e3,
        knots=4,
        freq_min=1.0,
        freq_max=20.0,
        sampling_freq=100.0,
        time_window=30,
        channels=3,
        name="Dispersion",
        *args,
        **kwargs
    ):
        super(Dispersion, self).__init__(name=name, **kwargs)
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.phase_velocity_min = phase_velocity_min
        self.phase_velocity_max = phase_velocity_max
        self.phase_velocity_std = phase_velocity_std
        self.knots = knots
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sampling_freq = sampling_freq
        self.time_window = time_window
        self.channels = channels

    def get_config(self):
        config = super(Dispersion, self).get_config()
        config.update(
            {
                "distance_min": self.distance_min,
                "distance_max": self.distance_max,
                "phase_velocity_min": self.phase_velocity_min,
                "phase_velocity_max": self.phase_velocity_max,
                "phase_velocity_std": self.phase_velocity_std,
                "knots": self.knots,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
                "sampling_freq": self.sampling_freq,
                "time_window": self.time_window,
                "channels": self.channels,
            }
        )
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.logf_knots = tf.linspace(
            tf.math.log(self.freq_min),
            tf.math.log(self.freq_max),
            self.knots,
        )

        self.wave_velocity_knots_ideal = self._linear_interpolate(
            self.logf_knots,
            tf.math.log(self.freq_min),
            tf.math.log(self.freq_max),
            self.phase_velocity_min,
            self.phase_velocity_max,
        )

        self.n_timesteps = int(self.time_window * self.sampling_freq)
        self.f = self._fftf(self.n_timesteps, self.sampling_freq)

        self.t = tf.linspace(
            0.0,
            self.time_window * ((self.n_timesteps - 1.0) / self.n_timesteps),
            self.n_timesteps,
        )

    def _fftf(self, n_timesteps, sampling_freq):
        """
        Similar to numpy fftfreq function but returning w.
        """
        n = n_timesteps
        d = 1.0 / sampling_freq
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
            stddev=self.phase_velocity_std,
            shape=[self.knots],
        )

        logf = tf.math.log(
            tf.maximum(tf.minimum(tf.abs(self.f), self.freq_max), self.freq_min)
        )

        wave_velocity_profile = cubic_interp1d(
            logf, self.logf_knots, wave_velocity_knots
        )

        return wave_velocity_profile

    def _disperse(self, x):
        r = tf.random.uniform(
            [],
            minval=self.distance_min,
            maxval=self.distance_max,
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

        phase_shifts = -2.0 * pi * self.f * delta_t
        phase_shifts = tf.expand_dims(phase_shifts, axis=1)

        x = tf.signal.ifft(xw * tf.math.exp(1j * tf.cast(phase_shifts, tf.complex128)))
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.cast(x, tf.float32)

        return x

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)

        return self._disperse(x)


@tf.keras.utils.register_keras_serializable()
class CrossCorrelateCircular(tf.keras.layers.Layer):
    def __init__(self, name="cross_correlate_circular", *args, **kwargs):
        super(CrossCorrelateCircular, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        shape_x = input_shape[0]
        self.batch_size = shape_x[0]
        self.n_timesteps = shape_x[1]
        self.n_channels = shape_x[2]
        self.w = self._fftw(self.n_timesteps)

    def call(self, input):
        a = input[0]
        b = input[1]

        shape = tf.shape(a)
        timesteps = shape[1]
        channels = shape[2]

        # Swap channels and timesteps axis.
        a = tf.transpose(a, perm=[0, 2, 1])
        b = tf.transpose(b, perm=[0, 2, 1])

        # Get mean of every trace of every channel.
        a = a - tf.reduce_mean(a, axis=2, keepdims=True)
        b = b - tf.reduce_mean(b, axis=2, keepdims=True)

        # Normalize each channel.
        eps = tf.constant(1e-37, dtype=tf.float32)
        a_l2 = tf.sqrt(tf.reduce_sum(tf.square(a), axis=2, keepdims=True))
        b_l2 = tf.sqrt(tf.reduce_sum(tf.square(b), axis=2, keepdims=True))
        a = a / (a_l2 + eps)
        b = b / (b_l2 + eps)

        # Cast to complex128 since fft with lower accuracy degrades performance.
        a = tf.cast(a, tf.complex128)
        b = tf.cast(b, tf.complex128)

        # Obtain frequency domain representation.
        aw = tf.signal.fft(a)
        bw = tf.signal.fft(b)

        # Cross correlate in frequency domain(multiplication is applied)
        cw = aw * tf.math.conj(bw)

        # Obtain correlation function in time domain.
        c = tf.signal.ifft(cw)

        # Cast back to lower precision and then center the correlation function.
        c = tf.cast(c, tf.float32)
        c = tf.roll(c, shift=(timesteps // 2), axis=2)

        # Swap channels and timesteps axis to restore.
        c = tf.transpose(c, perm=[0, 2, 1])

        return c

    @staticmethod
    def _get_shift_vector(w, step):
        return tf.exp(1j * step * w)

    @staticmethod
    def _fftw(n, d=1):
        """
        Similar to numpy fftfreq function but returning w.
        """
        f1 = tf.range(0, (n + 1) // 2) / (n * d)
        f2 = tf.range(-(n - 1) // 2, 0) / (n * d)
        f = tf.concat([f1, f2], axis=0)
        w = tf.constant(2 * pi, dtype=tf.float64) * f
        return tf.cast(w, tf.complex128)


@tf.keras.utils.register_keras_serializable()
class CrossCovarianceCircular(tf.keras.layers.Layer):
    def __init__(self, name="cross_covariance_circular", *args, **kwargs):
        super(CrossCovarianceCircular, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        shape_x = input_shape[0]
        self.batch_size = shape_x[0]
        self.n_timesteps = shape_x[1]
        self.n_channels = shape_x[2]
        self.w = self._fftw(self.n_timesteps)

    def call(self, input):
        a = input[0]
        b = input[1]

        shape = tf.shape(a)
        timesteps = shape[1]
        channels = shape[2]

        # Demean each channel over time axes.
        a = a - tf.reduce_mean(a, axis=(1), keepdims=True)
        b = b - tf.reduce_mean(b, axis=(1), keepdims=True)

        a = a / tf.sqrt(tf.cast(channels, tf.float32))
        b = b / tf.sqrt(tf.cast(channels, tf.float32))

        # Swap channels and timesteps axis.
        a = tf.transpose(a, perm=[0, 2, 1])
        b = tf.transpose(b, perm=[0, 2, 1])

        # Cast to complex128 since fft with lower accuracy degrades performance.
        a = tf.cast(a, tf.complex128)
        b = tf.cast(b, tf.complex128)

        # Obtain frequency domain representation.
        aw = tf.signal.fft(a)
        bw = tf.signal.fft(b)

        # Get cross covariance in frequency domain(multiplication is applied)
        cw = aw * tf.math.conj(bw)

        # Obtain cross covariation function in time domain.
        c = tf.signal.ifft(cw)

        # Cast back to lower precision and then center the cross covariance function.
        c = tf.cast(c, tf.float32)
        c = tf.roll(c, shift=(timesteps) // 2, axis=2)

        # Calculate mean cross variance by summing along channel.
        c = tf.reduce_sum(c, axis=1)

        return c

    @staticmethod
    def _get_shift_vector(w, step):
        return tf.exp(1j * step * w)

    @staticmethod
    def _fftw(n, d=1):
        """
        Similar to numpy fftfreq function but returning w.
        """
        f1 = tf.range(0, (n + 1) // 2) / (n * d)
        f2 = tf.range(-(n - 1) // 2, 0) / (n * d)
        f = tf.concat([f1, f2], axis=0)
        w = tf.constant(2 * pi, dtype=tf.float64) * f
        return tf.cast(w, tf.complex128)
