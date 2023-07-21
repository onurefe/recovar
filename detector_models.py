from config import BATCH_SIZE
from layers import CrossCovarianceCircular
import tensorflow as tf
from tensorflow import keras
from building_blocks import AutoencoderBlock30s, distance, latent_distance


@tf.keras.utils.register_keras_serializable()
class AutocovarianceDetector30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="autocovariance_detector_30s", *args, **kwargs):
        super(AutocovarianceDetector30s, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(AutocovarianceDetector30s, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.autoencoder = AutoencoderBlock30s("autoencoder")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        # Calculate standart deviation.
        # Because of the normalization during datagenerator.
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        x = x / (1e-20 + std)

        f, y = self.autoencoder(x, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class AutocovarianceDetectorDenoising30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="autocovariance_detector_denoising_30s", *args, **kwargs):
        super(AutocovarianceDetectorDenoising30s, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(AutocovarianceDetectorDenoising30s, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.autoencoder = AutoencoderBlock30s("autoencoder")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        # Calculate standart deviation.
        # Because of the normalization during datagenerator.
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        x = x / (1e-20 + std)

        if training:
            x_noised = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.3)
            std = tf.math.reduce_std(x_noised, axis=1, keepdims=True)
            x_noised = x_noised / (1e-20 + std)
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class AutocovarianceDetectorDenoisingRealisticNoise30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="autocovariance_detector_denoising_realistic_noise_30s",
        psd_ema=0.99,
        noise_std=0.3,
        *args,
        **kwargs
    ):
        self.psd_ema = psd_ema
        self.noise_std = noise_std
        self.pi = 3.141592653589793

        super(AutocovarianceDetectorDenoisingRealisticNoise30s, self).__init__(
            name=name, **kwargs
        )

    def get_config(self):
        config = super(
            AutocovarianceDetectorDenoisingRealisticNoise30s, self
        ).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.autoencoder = AutoencoderBlock30s("autoencoder")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def _normalize_std(self, x):
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        return x / (1e-20 + std)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self._normalize_std(x)

        if training:
            # Fourier transform.
            xw = tf.signal.fft(tf.cast(tf.transpose(x, perm=[0, 2, 1]), tf.complex64))

            # Get PSD.
            psd = tf.math.real(xw * tf.math.conj(xw))

            # Calculate mean over batch.
            psd = tf.reduce_mean(psd, axis=0, keepdims=True)

            # Generate random phase.
            phase = tf.random.uniform(
                shape=(BATCH_SIZE, self.N_CHANNELS, self.N_TIMESTEPS),
                minval=0,
                maxval=(self.pi * 2),
            )

            # Generate noise from PSD.
            noise = tf.signal.ifft(
                tf.cast(tf.sqrt(psd), tf.complex64)
                * tf.exp(1j * tf.cast(phase, tf.complex64))
            )

            # Make channels last axis and get real part of the signal.
            noise = tf.math.real(tf.transpose(noise, perm=[0, 2, 1]))

            # Normalize noise and return.
            noise = self._normalize_std(noise)

            # Generate noised signal.
            x_noised = x + noise * self.noise_std
            x_noised = self._normalize_std(x_noised)
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class Ensemble5CrossCovarianceDetector30s(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self, coeff=0.99, name="ensemble5_cross_covariance_detector30s", *args, **kwargs
    ):
        super(Ensemble5CrossCovarianceDetector30s, self).__init__(name=name, **kwargs)
        self.coeff = coeff

    def get_config(self):
        config = super(Ensemble5CrossCovarianceDetector30s, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.autoencoder1 = AutoencoderBlock30s("autoencoder1")
        self.autoencoder2 = AutoencoderBlock30s("autoencoder2")
        self.autoencoder3 = AutoencoderBlock30s("autoencoder3")
        self.autoencoder4 = AutoencoderBlock30s("autoencoder4")
        self.autoencoder5 = AutoencoderBlock30s("autoencoder5")

        self.cc = CrossCorrelateVectorsCircular()

        self.projector1 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="projector_1",
            trainable=True,
        )

        self.projector2 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="projector_2",
            trainable=True,
        )

        self.projector3 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="projector_3",
            trainable=True,
        )

        self.projector4 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="projector_4",
            trainable=True,
        )

        self.projector5 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="projector_5",
            trainable=True,
        )

        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn3 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn4 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn5 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def _ortho_loss(self, projector):
        return tf.reduce_mean(
            tf.abs(tf.matmul(projector, projector, transpose_b=True) - tf.eye(64))
        )

    def _mean_covariance(self, f1, f2, f3, f4, f5):
        fcov = (
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

        return fcov

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        x = x / (1e-20 + std)

        y1 = self.autoencoder1(x, training=training)
        y2 = self.autoencoder2(x, training=training)
        y3 = self.autoencoder3(x, training=training)
        y4 = self.autoencoder4(x, training=training)
        y5 = self.autoencoder5(x, training=training)

        _f1 = tf.stop_gradient(self.autoencoder1(x, training=training)[0])
        _f2 = tf.stop_gradient(self.autoencoder2(x, training=training)[0])
        _f3 = tf.stop_gradient(self.autoencoder3(x, training=training)[0])
        _f4 = tf.stop_gradient(self.autoencoder4(x, training=training)[0])
        _f5 = tf.stop_gradient(self.autoencoder5(x, training=training)[0])

        f1p = tf.transpose(
            tf.matmul(self.projector1, _f1, transpose_b=True), perm=[0, 2, 1]
        )
        f2p = tf.transpose(
            tf.matmul(self.projector2, _f2, transpose_b=True), perm=[0, 2, 1]
        )
        f3p = tf.transpose(
            tf.matmul(self.projector3, _f3, transpose_b=True), perm=[0, 2, 1]
        )
        f4p = tf.transpose(
            tf.matmul(self.projector4, _f4, transpose_b=True), perm=[0, 2, 1]
        )
        f5p = tf.transpose(
            tf.matmul(self.projector5, _f5, transpose_b=True), perm=[0, 2, 1]
        )

        f1p = self.bn1(f1p, training=training)
        f2p = self.bn2(f2p, training=training)
        f3p = self.bn3(f3p, training=training)
        f4p = self.bn4(f4p, training=training)
        f5p = self.bn5(f5p, training=training)

        # Orthonormality loss.
        ortho_loss = (
            self._ortho_loss(self.projector1)
            + self._ortho_loss(self.projector2)
            + self._ortho_loss(self.projector3)
            + self._ortho_loss(self.projector4)
            + self._ortho_loss(self.projector5)
        ) / 5.0

        # Projection pairwise distance loss.
        proj_dist_loss = (
            tf.reduce_mean(latent_distance(f1p, f2p), axis=0)
            + tf.reduce_mean(latent_distance(f1p, f3p), axis=0)
            + tf.reduce_mean(latent_distance(f2p, f3p), axis=0)
            + tf.reduce_mean(latent_distance(f1p, f4p), axis=0)
            + tf.reduce_mean(latent_distance(f2p, f4p), axis=0)
            + tf.reduce_mean(latent_distance(f3p, f4p), axis=0)
            + tf.reduce_mean(latent_distance(f1p, f5p), axis=0)
            + tf.reduce_mean(latent_distance(f2p, f5p), axis=0)
            + tf.reduce_mean(latent_distance(f3p, f5p), axis=0)
            + tf.reduce_mean(latent_distance(f4p, f5p), axis=0)
        ) / 10.0

        # Reconstruction loss.
        recon_loss = (
            tf.reduce_mean(distance(x, y1), axis=0)
            + tf.reduce_mean(distance(x, y2), axis=0)
            + tf.reduce_mean(distance(x, y3), axis=0)
            + tf.reduce_mean(distance(x, y4), axis=0)
            + tf.reduce_mean(distance(x, y5), axis=0)
        ) / 5.0

        self.add_loss(10 * ortho_loss + proj_dist_loss + recon_loss)

        return f1p, f2p, f3p, f4p, f5p, y1, y2, y3, y4, y5
