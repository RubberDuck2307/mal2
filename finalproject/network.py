import tensorflow as tf
from tensorflow.keras import layers, models


def vgg_block(num_convs, out_channels):
    block = tf.keras.Sequential()
    for _ in range(num_convs):
        block.add(layers.Conv2D(out_channels, kernel_size=3, padding="same", activation="gelu"))
    block.add(layers.BatchNormalization())
    block.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
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
        self.out.add(layers.Conv2D(128, kernel_size=2, padding="same", activation="gelu"))
        self.out.add(layers.Conv2D(64, kernel_size=2, padding="same", activation="gelu"))
        self.out.add(layers.Conv2D(1 * 5, kernel_size=2, padding="same", activation="sigmoid"))

        self.out_single = tf.keras.Sequential()
        self.out_single.add(layers.InputLayer(input_shape=(7, 7, 256)))
        self.out_single.add(layers.Flatten())
        self.out_single.add(layers.Dense(7 * 7 * 5, activation="gelu"))
        self.out_single.add(layers.Dense(5, activation="sigmoid"))
        self.out_single.add(layers.Reshape((1, 1, 5)))

    def call(self, x):
        x = self.conv_blks(x)
        logits = self.out_single(x)
        return logits
