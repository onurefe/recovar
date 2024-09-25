import tensorflow as tf
from tensorflow import keras
from seismic_purifier.config import BATCH_SIZE
from seismic_purifier.layers import AddNoise, NormalizeStd
from seismic_purifier.utils import demean, l2_normalize, l2_distance
from seismic_purifier.layers import (Downsample,
                                     Upsample,
                                     UpsampleNoactivation,
                                     ResIdentity,
                                     Padding)

@tf.keras.utils.register_keras_serializable()
class AutoencoderBlock(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="autoencoder_block", *args, **kwargs):
        super(AutoencoderBlock, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super(AutoencoderBlock, self).get_config()
        return config

    def build(self, input_shape=None):  # Create the state of the layer (weights)
        self.inp = keras.layers.InputLayer(
            (self.N_TIMESTEPS, self.N_CHANNELS), batch_size=BATCH_SIZE
        )

        self.down1 = Downsample(8, 15, name="down_1")  # 3000 -> 1500
        self.down2 = Downsample(16, 13, name="down_2")  # 1500 -> 750
        self.pad1 = Padding([1, 1])  # 750 -> 752
        self.down3 = Downsample(32, 11, name="down_3")  # 752 -> 376
        self.down4 = Downsample(64, 9, name="down_4")  # 376 -> 188
        self.down5 = Downsample(64, 7, name="down_5")  # 188 -> 94

        self.resid1 = ResIdentity(64, 5, name="resid_1")
        self.resid2 = ResIdentity(64, 5, name="resid_2")
        self.resid3 = ResIdentity(64, 5, name="resid_3")
        self.resid4 = ResIdentity(64, 5, name="resid_4")
        self.resid5 = ResIdentity(64, 5, name="resid_5")

        self.up1 = Upsample(32, 7, name="up_1")  # 94 -> 188
        self.up2 = Upsample(16, 9, name="up_2")  # 188 -> 376
        self.up3 = Upsample(8, 11, name="up_3")  # 376 -> 752
        self.crop1 = tf.keras.layers.Cropping1D(cropping=(1, 1))  # 752 -> 750
        self.up4 = Upsample(4, 13, name="up_4")  # 750 -> 1500
        self.up5 = UpsampleNoactivation(3, 15, name="up_5")

    def _encoder(self, x, training):
        x = self.down1(x, training=training)
        x = self.down2(x, training=training)
        x = self.pad1(x)
        x = self.down3(x, training=training)
        x = self.down4(x, training=training)
        x = self.down5(x, training=training)

        x = self.resid1(x, training=training)
        x = self.resid2(x, training=training)
        x = self.resid3(x, training=training)
        x = self.resid4(x, training=training)
        x = self.resid5(x, training=training)

        return x

    def _decoder(self, x, training):
        x = self.up1(x, training=training)
        x = self.up2(x, training=training)
        x = self.up3(x, training=training)
        x = self.crop1(x)
        x = self.up4(x, training=training)
        x = self.up5(x, training=training)

        return x

    def call(self, inputs, training=False):
        x = self.inp(inputs)
        x = tf.cast(x, dtype=tf.float32)

        f = self._encoder(x, training=training)
        y = self._decoder(f, training=training)

        return f, y
    

@tf.keras.utils.register_keras_serializable()
class RepresentationLearningSingleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(self, name="representation_learning_autoencoder", input_noise_std=1e-6, *args, **kwargs):
        super(RepresentationLearningSingleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std

    def get_config(self):
        config = super(RepresentationLearningSingleAutoencoder, self).get_config()
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
class RepresentationLearningDenoisingSingleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="representation_learning_denoising_autoencoder",
        input_noise_std=1e-6,
        denoising_noise_std=2e-1,
        *args,
        **kwargs
    ):
        super(RepresentationLearningDenoisingSingleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.denoising_noise_std = denoising_noise_std

    def get_config(self):
        config = super(RepresentationLearningDenoisingSingleAutoencoder, self).get_config()
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
            x_noised = self.normalize_denoising(self.add_noise_denoising(x))
        else:
            x_noised = x

        f, y = self.autoencoder(x_noised, training=training)
        f = self.bn(f, training=training)

        self.add_loss(tf.reduce_mean(l2_distance(x, y), axis=0))

        return f, y


@tf.keras.utils.register_keras_serializable()
class RepresentationLearningMultipleAutoencoder(keras.Model):
    N_TIMESTEPS = 3000
    N_CHANNELS = 3

    def __init__(
        self,
        name="representation_learning_autoencoder_ensemble",
        input_noise_std=1e-6,
        eps=1e-27,
        *args,
        **kwargs
    ):
        super(RepresentationLearningMultipleAutoencoder, self).__init__(name=name, **kwargs)
        self.input_noise_std = input_noise_std
        self.eps = eps

    def get_config(self):
        config = super(RepresentationLearningMultipleAutoencoder, self).get_config()
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
