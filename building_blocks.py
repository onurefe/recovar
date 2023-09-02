from config import BATCH_SIZE
from layers import (
    Downsample,
    Upsample,
    UpsampleNoactivation,
    ResIdentity,
    Padding,
)
import tensorflow as tf
from tensorflow import keras

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
