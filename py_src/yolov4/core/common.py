import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.tanh(tf.math.log(1 + tf.exp(inputs)))


class YOLOConv2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
        bn: bool = True,
        activation: str = "mish",
        **kwargs
    ):
        super(YOLOConv2D, self).__init__(**kwargs)
        self.sequential = tf.keras.Sequential()
        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if strides == 2:
            self.sequential.add(layers.ZeroPadding2D(((1, 0), (1, 0))))

        self.sequential.add(
            layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same" if strides == 1 else "valid",
                strides=strides,
                use_bias=not bn,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
            )
        )

        if bn:
            self.sequential.add(layers.BatchNormalization())

        if activation == "mish":
            self.sequential.add(Mish())
        elif activation == "leaky":
            self.sequential.add(layers.LeakyReLU(alpha=0.1))
        else:
            self.sequential.add(layers.ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)


def convolutional(
    input_layer,
    filters_shape,
    downsample=False,
    activate=True,
    bn=True,
    activate_type="leaky",
):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(
            input_layer
        )
        padding = "valid"
        strides = 2
    else:
        strides = 1
        padding = "same"

    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
    )(input_layer)

    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = Mish()(conv)

    return conv
