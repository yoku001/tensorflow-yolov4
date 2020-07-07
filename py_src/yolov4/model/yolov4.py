"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from tensorflow.keras import layers, Model

from .common import YOLOConv2D
from .backbone import CSPDarknet53


class YOLOv4(Model):
    """
    Path Aggregation Network(PAN)
    Spatial Attention Module(SAM)
    Bounding Box(BBox)
    """

    def __init__(self, num_classes: int):
        super(YOLOv4, self).__init__(name="YOLOv4")
        self.csp_darknet53 = CSPDarknet53()

        self.conv78 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.upSampling78 = layers.UpSampling2D()
        self.conv79 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.concat78_79 = layers.Concatenate(axis=-1)

        self.conv80 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.conv81 = YOLOConv2D(filters=512, kernel_size=3, activation="leaky")
        self.conv82 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.conv83 = YOLOConv2D(filters=512, kernel_size=3, activation="leaky")
        self.conv84 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")

        self.conv85 = YOLOConv2D(filters=128, kernel_size=1, activation="leaky")
        self.upSampling85 = layers.UpSampling2D()
        self.conv86 = YOLOConv2D(filters=128, kernel_size=1, activation="leaky")
        self.concat85_86 = layers.Concatenate(axis=-1)

        self.conv87 = YOLOConv2D(filters=128, kernel_size=1, activation="leaky")
        self.conv88 = YOLOConv2D(filters=256, kernel_size=3, activation="leaky")
        self.conv89 = YOLOConv2D(filters=128, kernel_size=1, activation="leaky")
        self.conv90 = YOLOConv2D(filters=256, kernel_size=3, activation="leaky")
        self.conv91 = YOLOConv2D(filters=128, kernel_size=1, activation="leaky")

        self.conv92 = YOLOConv2D(filters=256, kernel_size=3, activation="leaky")
        self.conv93 = YOLOConv2D(
            filters=3 * (num_classes + 5), kernel_size=1, activation=None,
        )

        self.conv94 = YOLOConv2D(
            filters=256, kernel_size=3, strides=2, activation="leaky"
        )
        self.concat84_94 = layers.Concatenate(axis=-1)

        self.conv95 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.conv96 = YOLOConv2D(filters=512, kernel_size=3, activation="leaky")
        self.conv97 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")
        self.conv98 = YOLOConv2D(filters=512, kernel_size=3, activation="leaky")
        self.conv99 = YOLOConv2D(filters=256, kernel_size=1, activation="leaky")

        self.conv100 = YOLOConv2D(
            filters=512, kernel_size=3, activation="leaky"
        )
        self.conv101 = YOLOConv2D(
            filters=3 * (num_classes + 5), kernel_size=1, activation=None,
        )

        self.conv102 = YOLOConv2D(
            filters=512, kernel_size=3, strides=2, activation="leaky"
        )
        self.concat77_102 = layers.Concatenate(axis=-1)

        self.conv103 = YOLOConv2D(
            filters=512, kernel_size=1, activation="leaky"
        )
        self.conv104 = YOLOConv2D(
            filters=1024, kernel_size=3, activation="leaky"
        )
        self.conv105 = YOLOConv2D(
            filters=512, kernel_size=1, activation="leaky"
        )
        self.conv106 = YOLOConv2D(
            filters=1024, kernel_size=3, activation="leaky"
        )
        self.conv107 = YOLOConv2D(
            filters=512, kernel_size=1, activation="leaky"
        )

        self.conv108 = YOLOConv2D(
            filters=1024, kernel_size=3, activation="leaky"
        )
        self.conv109 = YOLOConv2D(
            filters=3 * (num_classes + 5), kernel_size=1, activation=None,
        )

    def call(self, x):
        """
        @param x: Dim(batch, input_size, input_size, channels)
            The element has a value between 0.0 and 1.0.

        @return (s_pred, m_pred, l_pred)
            downsampling_size = [8, 16, 32]
            output_size = input_size // downsampling_szie
            s_pred = Dim(batch, output_size[0], output_size[0],
                                        num_anchors * (5 + num_classes))
            l_pred = Dim(batch, output_size[1], output_size[1],
                                        num_anchors * (5 + num_classes))
            m_pred = Dim(batch, output_size[2], output_size[2],
                                        num_anchors * (5 + num_classes))

        Ref: https://arxiv.org/abs/1612.08242 - YOLOv2

        5 + num_classes = (t_x, t_y, t_w, t_h, t_o, c_0, c_1, ...)

        A top-left coordinate of a grid is (c_x, c_y).
        A dimension prior is (p_w, p_h).(== anchor size)
                [[[12, 16],   [19, 36],   [40, 28]  ],
                 [[36, 75],   [76, 55],   [72, 146] ],
                 [[142, 110], [192, 243], [459, 401]]]
        Pr == Probability.

        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)
        sigmoid(t_o) == confidence == Pr(Object) ∗ IoU(b, Object)
        sigmoid(c_i) == conditional class probability == Pr(Class_i|Object)
        """
        route1, route2, route3 = self.csp_darknet53(x)

        x1 = self.conv78(route3)
        part2 = self.upSampling78(x1)
        part1 = self.conv79(route2)
        x1 = self.concat78_79([part1, part2])

        x1 = self.conv80(x1)
        x1 = self.conv81(x1)
        x1 = self.conv82(x1)
        x1 = self.conv83(x1)
        x1 = self.conv84(x1)

        x2 = self.conv85(x1)
        part2 = self.upSampling85(x2)
        part1 = self.conv86(route1)
        x2 = self.concat85_86([part1, part2])

        x2 = self.conv87(x2)
        x2 = self.conv88(x2)
        x2 = self.conv89(x2)
        x2 = self.conv90(x2)
        x2 = self.conv91(x2)

        s_pred = self.conv92(x2)
        s_pred = self.conv93(s_pred)

        x2 = self.conv94(x2)
        x2 = self.concat84_94([x2, x1])

        x2 = self.conv95(x2)
        x2 = self.conv96(x2)
        x2 = self.conv97(x2)
        x2 = self.conv98(x2)
        x2 = self.conv99(x2)

        m_pred = self.conv100(x2)
        m_pred = self.conv101(m_pred)

        x2 = self.conv102(x2)
        x2 = self.concat77_102([x2, route3])

        x2 = self.conv103(x2)
        x2 = self.conv104(x2)
        x2 = self.conv105(x2)
        x2 = self.conv106(x2)
        x2 = self.conv107(x2)

        l_pred = self.conv108(x2)
        l_pred = self.conv109(l_pred)

        return s_pred, m_pred, l_pred
