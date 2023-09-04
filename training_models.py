from config import BATCH_SIZE
import tensorflow as tf
from tensorflow import keras
import numpy as np
from layers import AddNoise, NormalizeStd
from building_blocks import AutoencoderBlock
from utils import demean, l2_normalize, l2_distance


@tf.keras.utils.register_keras_serializable()
class Autoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="autoencoder", input_noise_std=1e-6, *args, **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.autoencoder = AutoencoderBlock("autoencoder_block")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f, y = self.autoencoder(x, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(l2_distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class DenoisingAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="denoising_autoencoder",
        input_noise_std=1e-6,
        denoising_noise_std=2e-1,
        *args,
        **kwargs
    ):
        super(DenoisingAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.denoising_noise_std = denoising_noise_std

    def get_config(self):
        config = super(DenoisingAutoencoder, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise_input = AddNoise(
            name="add_noise_input", stddev=self.input_noise_std
        )
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.add_noise_denoising = AddNoise(
            name="add_noise_denoising", stddev=self.denoising_noise_std
        )
        self.normalize_denoising = NormalizeStd(name="normalize_denoising")

        self.autoencoder = AutoencoderBlock("autoencoder_block")
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise_input(x)
        x = self.normalize2(x)

        if training:
            x_noised = self.add_noise_denoising(x)
            x_noised = self.normalize_denoising(x)
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(l2_distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class AutoencoderEnsemble(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="autoencoder_ensemble",
        input_noise_std=1e-6,
        eps=1e-27,
        *args,
        **kwargs
    ):
        super(AutoencoderEnsemble, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.eps = eps

    def get_config(self):
        config = super(AutoencoderEnsemble, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.normalize1 = NormalizeStd(name="normalize_1")
        self.add_noise = AddNoise(stddev=self.input_noise_std)
        self.normalize2 = NormalizeStd(name="normalize_2")

        self.autoencoder1 = AutoencoderBlock("autoencoder_block1")
        self.autoencoder2 = AutoencoderBlock("autoencoder_block2")
        self.autoencoder3 = AutoencoderBlock("autoencoder_block3")
        self.autoencoder4 = AutoencoderBlock("autoencoder_block4")
        self.autoencoder5 = AutoencoderBlock("autoencoder_block5")

        self.linear1 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_1",
            trainable=True,
        )

        self.linear2 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_2",
            trainable=True,
        )

        self.linear3 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_3",
            trainable=True,
        )

        self.linear4 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_4",
            trainable=True,
        )

        self.linear5 = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(64, 64)),
            dtype=tf.float32,
            name="linear_5",
            trainable=True,
        )

        self.bn1 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn3 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn4 = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.bn5 = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        x = self.normalize1(x)
        x = self.add_noise(x)
        x = self.normalize2(x)

        f1, y1 = self.autoencoder1(x, training=training)
        f2, y2 = self.autoencoder2(x, training=training)
        f3, y3 = self.autoencoder3(x, training=training)
        f4, y4 = self.autoencoder4(x, training=training)
        f5, y5 = self.autoencoder5(x, training=training)

        f1p = tf.transpose(
            tf.matmul(self.linear1, tf.stop_gradient(f1), transpose_b=True),
            perm=[0, 2, 1],
        )
        f2p = tf.transpose(
            tf.matmul(self.linear2, tf.stop_gradient(f2), transpose_b=True),
            perm=[0, 2, 1],
        )
        f3p = tf.transpose(
            tf.matmul(self.linear3, tf.stop_gradient(f3), transpose_b=True),
            perm=[0, 2, 1],
        )
        f4p = tf.transpose(
            tf.matmul(self.linear4, tf.stop_gradient(f4), transpose_b=True),
            perm=[0, 2, 1],
        )
        f5p = tf.transpose(
            tf.matmul(self.linear5, tf.stop_gradient(f5), transpose_b=True),
            perm=[0, 2, 1],
        )

        f1p = self.bn1(f1p, training=training)
        f2p = self.bn2(f2p, training=training)
        f3p = self.bn3(f3p, training=training)
        f4p = self.bn4(f4p, training=training)
        f5p = self.bn5(f5p, training=training)

        reconstruction_loss = self._get_reconstruction_loss(x, y1, y2, y3, y4, y5)
        ensemble_distance_loss = self._get_ensemble_distance_loss(
            f1p, f2p, f3p, f4p, f5p
        )

        self.add_loss(reconstruction_loss + ensemble_distance_loss)

        return f1p, f2p, f3p, f4p, f5p, y1, y2, y3, y4, y5

    def _get_ensemble_distance_loss(self, f1p, f2p, f3p, f4p, f5p):
        ensemble_distance_loss = (
            tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f2p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f3p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f3p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f3p, f4p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f1p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f2p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f3p, f5p), axis=0)
            + tf.reduce_mean(self._l2_after_channel_normalizing(f4p, f5p), axis=0)
        ) / 10.0

        return ensemble_distance_loss

    def _get_reconstruction_loss(self, x, y1, y2, y3, y4, y5):
        reconstruction_loss = (
            tf.reduce_mean(l2_distance(x, y1), axis=0)
            + tf.reduce_mean(l2_distance(x, y2), axis=0)
            + tf.reduce_mean(l2_distance(x, y3), axis=0)
            + tf.reduce_mean(l2_distance(x, y4), axis=0)
            + tf.reduce_mean(l2_distance(x, y5), axis=0)
        ) / 5.0

        return reconstruction_loss

    def _l2_after_channel_normalizing(self, x, y):
        x = demean(x, axis=1)
        y = demean(y, axis=1)

        x = l2_normalize(x, axis=1)
        y = l2_normalize(y, axis=1)

        distance = tf.sqrt(tf.reduce_mean(tf.square(x - y), axis=[1, 2]))

        return distance
