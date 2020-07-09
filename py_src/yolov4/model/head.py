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
import tensorflow as tf
from tensorflow.keras import activations, backend, layers, Model


class YOLOv3Head(Model):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3Head, self).__init__(name="YOLOv3Head")
        self.anchors = anchors
        self.grid_coord = []
        self.grid_size = None
        self.image_size = None
        self.num_classes = num_classes
        self.scales = xysclaes

        self.reshape0 = layers.Reshape((-1,))
        self.reshape1 = layers.Reshape((-1,))
        self.reshape2 = layers.Reshape((-1,))

        self.concat0 = layers.Concatenate(axis=-1)

    def build(self, input_shape):
        grid = (input_shape[0][1], input_shape[1][1], input_shape[2][1])

        self.reshape0.target_shape = (grid[0], grid[0], 3, 5 + self.num_classes)
        self.reshape1.target_shape = (grid[1], grid[1], 3, 5 + self.num_classes)
        self.reshape2.target_shape = (grid[2], grid[2], 3, 5 + self.num_classes)

        for i in range(3):
            xy_grid = tf.meshgrid(tf.range(grid[i]), tf.range(grid[i]))
            xy_grid = tf.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[tf.newaxis, :, :, tf.newaxis, :]
            xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)
            self.grid_coord.append(xy_grid)

        self.grid_size = grid
        self.image_size = grid[0] * 8

    def call(self, x):
        raw_s, raw_m, raw_l = x

        raw_s = self.reshape0(raw_s)
        raw_m = self.reshape1(raw_m)
        raw_l = self.reshape2(raw_l)

        pred = []
        for i, raw_pred in enumerate((raw_s, raw_m, raw_l)):
            txty, twth, raw_conf, raw_prob = tf.split(
                raw_pred, (2, 2, 1, self.num_classes), axis=-1
            )
            txty = (activations.sigmoid(txty) - 0.5) * self.scales[i] + 0.5
            bxby = (txty + self.grid_coord[i]) / self.grid_size[i]

            bwbh = (self.anchors[i] / self.image_size) * backend.exp(twth)

            conf = activations.sigmoid(raw_conf)
            prob = activations.sigmoid(raw_prob)

            pred.append(self.concat0([bxby, bwbh, conf, prob]))

        return pred
