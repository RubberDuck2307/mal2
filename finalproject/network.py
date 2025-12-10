import tensorflow as tf
from tensorflow.keras import layers, models


class VGGBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, out_channels, dropout_rate=0.2):
        super().__init__()
        self.convs = []

        for _ in range(num_convs):
            self.convs.append(layers.Conv2D(out_channels, kernel_size=3, padding="same"))
            self.convs.append(layers.BatchNormalization())
            self.convs.append(layers.LeakyReLU())
            self.convs.append(layers.Dropout(dropout_rate))

        self.res = layers.Conv2D(out_channels, kernel_size=1, padding="same")
        self.activation = layers.LeakyReLU()

        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

    def call(self, x, training=False):
        identity = x
        for layer in self.convs:
            x = layer(x, training=training)

        x += self.res(identity)
        x = self.activation(x)
        x = self.pool(x)
        return x


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_blks = models.Sequential([
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            VGGBlock(2, 32),
            VGGBlock(2, 64),
            VGGBlock(2, 128),
            VGGBlock(2, 256),
        ])

        self.out = tf.keras.Sequential([
            layers.Conv2D(128, kernel_size=6, padding="valid", activation="gelu"),
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=2, padding="valid", activation="gelu"),
            layers.Dropout(0.2),
            layers.Conv2D(1 * 4, kernel_size=2, padding="valid", activation="sigmoid")
        ])

        self.out_single = tf.keras.Sequential()
        self.out_single.add(layers.InputLayer(input_shape=(8, 8, 256)))
        self.out_single.add(layers.Flatten())
        self.out_single.add(layers.Dense(4 * 4 * 4, activation="gelu"))
        self.out_single.add(layers.Dense(2 * 2 * 4, activation="gelu"))
        self.out_single.add(layers.Dense(4, activation="sigmoid"))
        self.out_single.add(layers.Reshape((1, 1, 4)))

    def build(self, input_shape):
        self.conv_blks.build(input_shape)
        x = self.conv_blks.compute_output_shape(input_shape)
        self.out.build(x)
        super(Net, self).build(input_shape)

    def call(self, x):
        x = self.conv_blks(x)
        logits = self.out_single(x)
        logits = tf.squeeze(logits)
        return logits
