import tensorflow as tf
from tensorflow.keras import layers, models


def vgg_block(num_convs, out_channels, dropout_rate =0.2):
    block = tf.keras.Sequential()
    for _ in range(num_convs):
        block.add(layers.Conv2D(out_channels, kernel_size=3, padding="same", activation="gelu"))
    block.add(layers.BatchNormalization())
    block.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    block.add(layers.Dropout(dropout_rate))
    return block


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_blks = models.Sequential([
            vgg_block(2, 256),
            vgg_block(2, 256),
            vgg_block(2, 256),
            vgg_block(2, 256),
            vgg_block(2, 256),
        ])

        self.out = tf.keras.Sequential()
        self.out = tf.keras.Sequential([
            layers.Conv2D(128, kernel_size=6, padding="valid", activation="gelu"),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=2, padding="valid", activation="gelu"),
            layers.Dropout(0.2),
            layers.Conv2D(1 * 5, kernel_size=2, padding="valid", activation="sigmoid")
        ])
        #
        # self.out_single = tf.keras.Sequential()
        # self.out_single.add(layers.InputLayer(input_shape=(7, 7, 256)))
        # self.out_single.add(layers.Flatten())
        # self.out_single.add(layers.Dense(7 * 7 * 5, activation="gelu"))
        # self.out_single.add(layers.Dense(7 * 7 * 5, activation="gelu"))
        # self.out_single.add(layers.Dense(5, activation="sigmoid"))
        # self.out_single.add(layers.Reshape((1, 1, 5)))

    def build(self, input_shape):
        self.conv_blks.build(input_shape)
        x = self.conv_blks.compute_output_shape(input_shape)
        self.out.build(x)
        super(Net, self).build(input_shape)

    def call(self, x):
        x = self.conv_blks(x)
        logits = self.out(x)
        return logits
