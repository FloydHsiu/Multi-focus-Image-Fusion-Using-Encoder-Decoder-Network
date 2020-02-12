import tensorflow as tf
from tensorflow import keras as k
from util_tf import instance_norm


def MFNet():
    # Model initialization
    inputs1 = k.Input(shape=(None, None, 1))
    inputs2 = k.Input(shape=(None, None, 1))
    concat = k.layers.concatenate([inputs1, inputs2], axis=3)
    cnn = mf_module(name='mf_module')
    outputs = tf.nn.tanh(cnn(concat))
    cnn = k.Model(
        inputs=[inputs1, inputs2],
        outputs=outputs, name='MFNet')
    return cnn


class mf_module(k.Model):
    def __init__(self, name):
        super(mf_module, self).__init__(name=name)
        self.initializer = k.initializers.TruncatedNormal(stddev=0.02)
        self.regularizer = tf.keras.regularizers.l1(l=0.0005)

        self.conv1 = k.layers.Conv2D(32, (7, 7),
                                     strides=(1, 1),
                                     padding='valid', use_bias=True,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     input_shape=(None, None, 2))

        self.conv2 = k.layers.Conv2D(64, (4, 4),
                                     strides=(2, 2),
                                     padding='valid', use_bias=False,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer)

        self.norm2 = instance_norm()

        self.conv2_2 = k.layers.Conv2D(64, (3, 3),
                                       strides=(1, 1),
                                       padding='valid', use_bias=False,
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=self.regularizer)

        self.norm2_2 = instance_norm()

        self.conv3 = k.layers.Conv2D(128, (4, 4),
                                     strides=(2, 2),
                                     padding='valid', use_bias=False,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer)

        self.norm3 = instance_norm()

        self.conv3_2 = k.layers.Conv2D(128, (3, 3),
                                       strides=(1, 1),
                                       padding='valid', use_bias=False,
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=self.regularizer)

        self.norm3_2 = instance_norm()

        self.conv4 = k.layers.Conv2D(256, (4, 4),
                                     strides=(2, 2),
                                     padding='valid', use_bias=False,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer)

        self.norm4 = instance_norm()

        self.conv4_2 = k.layers.Conv2D(256, (3, 3),
                                       strides=(1, 1),
                                       padding='valid', use_bias=False,
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=self.regularizer)

        self.norm4_2 = instance_norm()

        self.upsample5 = k.layers.UpSampling2D()
        self.conv5 = k.layers.Conv2D(
            128, (3, 3),
            strides=(1, 1),
            padding='valid', use_bias=False, kernel_initializer=self.
            initializer, kernel_regularizer=self.regularizer)
        self.norm5 = instance_norm()

        self.conv5_2 = k.layers.Conv2D(128, (3, 3),
                                       strides=(1, 1),
                                       padding='valid', use_bias=False,
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=self.regularizer)
        self.norm5_2 = instance_norm()

        self.upsample6 = k.layers.UpSampling2D()
        self.conv6 = k.layers.Conv2D(
            64, (3, 3),
            strides=(1, 1),
            padding='valid', use_bias=False, kernel_initializer=self.
            initializer, kernel_regularizer=self.regularizer)
        self.norm6 = instance_norm()

        self.conv6_2 = k.layers.Conv2D(64, (3, 3),
                                       strides=(1, 1),
                                       padding='valid', use_bias=False,
                                       kernel_initializer=self.initializer,
                                       kernel_regularizer=self.regularizer)
        self.norm6_2 = instance_norm()

        self.upsample7 = k.layers.UpSampling2D()
        self.conv7 = k.layers.Conv2D(
            32, (3, 3),
            strides=(1, 1),
            padding='valid', use_bias=False, kernel_initializer=self.
            initializer, kernel_regularizer=self.regularizer)
        self.norm7 = instance_norm()

        self.conv8 = k.layers.Conv2D(
            1, (7, 7),
            strides=(1, 1),
            padding='valid', use_bias=True, kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)

    def call(self, inputs, training=False):
        # Encoder
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.conv1(x))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm2(self.conv2(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm2_2(self.conv2_2(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm3(self.conv3(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm3_2(self.conv3_2(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm4(self.conv4(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm4_2(self.conv4_2(x), training=training))

        # Decoder
        x = self.upsample5(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm5(self.conv5(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm5_2(self.conv5_2(x), training=training))
        x = self.upsample6(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm6(self.conv6(x), training=training))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm6_2(self.conv6_2(x), training=training))
        x = self.upsample7(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        x = tf.nn.relu(self.norm7(self.conv7(x), training=training))
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = self.conv8(x)
        return x
