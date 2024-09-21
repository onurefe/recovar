import tensorflow as tf
from math import pi

@tf.keras.utils.register_keras_serializable()
class AddNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, name="add_noise", *args, **kwargs):
        super(AddNoise, self).__init__(name=name, **kwargs)
        self.stddev = stddev

    def call(self, x):
        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=self.stddev)
        return x


@tf.keras.utils.register_keras_serializable()
class NormalizeStd(tf.keras.layers.Layer):
    def __init__(self, axis=1, eps=1e-27, name="normalize_std", *args, **kwargs):
        super(NormalizeStd, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.eps = eps

    def call(self, x):
        std = tf.math.reduce_std(x, axis=self.axis, keepdims=True)
        x = x / (self.eps + std)
        return x


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
        c = tf.math.real(c)
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
