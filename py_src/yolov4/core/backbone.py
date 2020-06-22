import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Layer
from . import common


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
            self.sequential.add(layers.LeakyReLU())
        else:
            self.sequential.add(layers.ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)


class _ResBlock(Model):
    def __init__(self, filters_1: int, filters_2: int):
        super(_ResBlock, self).__init__()
        self.sequential = tf.keras.Sequential()
        self.sequential.add(
            YOLOConv2D(filters=filters_1, kernel_size=1, strides=1)
        )
        self.sequential.add(
            YOLOConv2D(filters=filters_2, kernel_size=3, strides=1)
        )

    def call(self, x):
        ret = self.sequential(x)
        x = x + ret
        return x


class ResBlock(Model):
    def __init__(self, filters_1: int, filters_2: int, iteration: int):
        super(ResBlock, self).__init__()
        self.iteration = iteration
        self.sequential = tf.keras.Sequential()
        for _ in range(self.iteration):
            self.sequential.add(
                _ResBlock(filters_1=filters_1, filters_2=filters_2)
            )

    def call(self, x):
        return self.sequential(x)


class CSPResNet(Model):
    def __init__(self, filters_1: int, filters_2: int, iteration: int):
        super(CSPResNet, self).__init__()
        self.pre_conv = YOLOConv2D(filters=filters_1, kernel_size=3, strides=2)

        # Do not change the order of declaration
        self.part2_conv = YOLOConv2D(
            filters=filters_2, kernel_size=1, strides=1,
        )

        self.part1_conv1 = YOLOConv2D(
            filters=filters_2, kernel_size=1, strides=1
        )
        self.part1_res_block = ResBlock(
            filters_1=filters_1 // 2, filters_2=filters_2, iteration=iteration
        )
        self.part1_conv2 = YOLOConv2D(
            filters=filters_2, kernel_size=1, strides=1
        )

        self.post_conv = YOLOConv2D(filters=filters_1, kernel_size=1, strides=1)

    def call(self, x):
        x = self.pre_conv(x)

        part2 = self.part2_conv(x)

        part1 = self.part1_conv1(x)
        part1 = self.part1_res_block(part1)
        part1 = self.part1_conv2(part1)

        x = tf.concat([part1, part2], axis=-1)

        x = self.post_conv(x)
        return x


class SPP(Model):
    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = tf.keras.layers.MaxPooling2D(
            (13, 13), strides=1, padding="same"
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(
            (9, 9), strides=1, padding="same"
        )
        self.pool3 = tf.keras.layers.MaxPooling2D(
            (5, 5), strides=1, padding="same"
        )

    def call(self, x):
        return tf.concat([self.pool1(x), self.pool2(x), self.pool3(x), x], -1)


class CSPDarknet53(Model):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv1 = YOLOConv2D(filters=32, kernel_size=3, strides=1,)
        self.res_block1 = CSPResNet(filters_1=64, filters_2=64, iteration=1)
        self.res_block2 = CSPResNet(filters_1=128, filters_2=64, iteration=2)
        self.res_block3 = CSPResNet(filters_1=256, filters_2=128, iteration=8)

        self.res_block4 = CSPResNet(filters_1=512, filters_2=256, iteration=8)

        self.res_block5 = CSPResNet(filters_1=1024, filters_2=512, iteration=4)
        self.conv2 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv3 = YOLOConv2D(
            filters=1024, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv4 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )
        self.spp = SPP()
        self.conv5 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv6 = YOLOConv2D(
            filters=1024, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv7 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        route1 = x

        x = self.res_block4(x)

        route2 = x

        x = self.res_block5(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.spp(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        route3 = x

        return (route1, route2, route3)
