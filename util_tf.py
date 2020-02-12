import tensorflow as tf
from tensorflow import keras as k


class instance_norm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(instance_norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.epsilon = 1e-6
        self.scale = self.add_weight(
            name="scale", shape=[input_shape[-1]],
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=True)
        self.offset = self.add_weight(name="offset",
                                      shape=[input_shape[-1]],
                                      initializer='zeros',
                                      trainable=True)
        super(instance_norm, self).build(input_shape)

    def call(self, input):
        mean, var = tf.nn.moments(input, [1, 2], keepdims=True)
        out = self.scale * \
            tf.divide(input - mean, tf.sqrt(var+self.epsilon)) + self.offset
        return out

    def compute_output_shape(self, input_shape):
        return input_shape
