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

import configparser
import time
import numpy as np
import cv2
import tensorflow as tf

from .core import utils
from .core import yolov4

parser = configparser.ConfigParser()


class YoloV4:
    def __init__(self):
        self.strides = np.array([8, 16, 32])
        self.anchors = np.array(
            [
                12,
                16,
                19,
                36,
                40,
                28,
                36,
                75,
                76,
                55,
                72,
                146,
                142,
                110,
                192,
                243,
                459,
                401,
            ],
            dtype=np.float32,
        ).reshape(3, 3, 2)
        self.xyscale = np.array([1.2, 1.1, 1.05])
        self.width = self.height = 608

    def inference(self, names_path, media_path, weights_path, isImage=True):
        classes = utils.read_class_names(names_path)
        num_class = len(classes)

        self.make_model(num_class)
        self.load_weights(weights_path)

        if isImage:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = self.predict(frame, classes)

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            cv2.waitKey(0)
        else:
            vid = cv2.VideoCapture(media_path)
            while True:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("No image! Try with another video format")

                prev_time = time.time()
                image = self.predict(frame, classes)
                curr_time = time.time()
                exec_time = curr_time - prev_time

                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                print(info)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def train(self):
        pass

    def make_model(self, num_class):
        input_layer = tf.keras.layers.Input([self.height, self.width, 3])

        feature_maps = yolov4.YOLOv4(input_layer, num_class)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = yolov4.decode(fm, num_class, i)
            bbox_tensors.append(bbox_tensor)
        self.model = tf.keras.Model(input_layer, bbox_tensors)

    def load_weights(self, weights_path):
        if (
            weights_path.split(".")[len(weights_path.split(".")) - 1]
            == "weights"
        ):
            utils.load_weights(self.model, weights_path)
        else:
            self.model.load_weights(weights_path).expect_partial()

        self.model.summary()

    def predict(self, frame, classes):
        frame_size = frame.shape[:2]

        image_data = utils.image_preporcess(
            np.copy(frame), [self.height, self.width]
        )
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)

        pred_bbox = utils.postprocess_bbbox(
            pred_bbox, self.anchors, self.strides, self.xyscale
        )
        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, self.width, 0.25
        )
        bboxes = utils.nms(bboxes, 0.213, method="nms")

        return utils.draw_bbox(frame, bboxes, classes)
