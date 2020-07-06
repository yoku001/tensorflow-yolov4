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
from os import path
import time
from typing import Union

import cv2
import numpy as np
import tensorflow as tf

from ..utility import dataset, media, predict, train, weights
from ..model import yolov4


class YOLOv4:
    def __init__(self):
        """
        Default configuration
        """
        self.anchors = [
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
        ]
        self._classes = None
        self._has_weights = False
        self.input_size = 608
        self.model = None
        self.strides = [8, 16, 32]
        self.xyscales = [1.2, 1.1, 1.05]

    @property
    def anchors(self):
        """
        Usage:
            yolo.anchors = [12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
            yolo.anchors = np.array([12, 16, 19, 36, 40, 28, 36, 75,
                            76, 55, 72, 146, 142, 110, 192, 243, 459, 401])
            print(yolo.anchors)
        """
        return self._anchors

    @anchors.setter
    def anchors(self, anchors: Union[list, tuple, np.ndarray]):
        if isinstance(anchors, (list, tuple)):
            self._anchors = np.array(anchors)
        elif isinstance(anchors, np.ndarray):
            self._anchors = anchors

        self._anchors = self._anchors.astype(np.float32).reshape(3, 3, 2)

    @property
    def classes(self):
        """
        Usage:
            yolo.classes = {0: 'person', 1: 'bicycle', 2: 'car', ...}
            yolo.classes = "path/classes"
            print(len(yolo.classes))
        """
        return self._classes

    @classes.setter
    def classes(self, data: Union[str, dict]):
        if isinstance(data, str):
            self._classes = media.read_classes_names(data)
        elif isinstance(data, dict):
            self._classes = data
        else:
            raise TypeError("YOLOv4: Set classes path or dictionary")

    @property
    def input_size(self):
        """
        Usage:
            yolo.input_size = 608
            print(yolo.input_size)
        """
        return self._input_size

    @input_size.setter
    def input_size(self, size: int):
        if size % 32 == 0:
            self._input_size = size
        else:
            raise ValueError("YOLOv4: Set input_size to multiples of 32")

    @property
    def strides(self):
        """
        Usage:
            yolo.strides = [8, 16, 32]
            yolo.strides = np.array([8, 16, 32])
            print(yolo.strides)
        """
        return self._strides

    @strides.setter
    def strides(self, strides: Union[list, tuple, np.ndarray]):
        if isinstance(strides, (list, tuple)):
            self._strides = np.array(strides)
        elif isinstance(strides, np.ndarray):
            self._strides = strides

    @property
    def xyscales(self):
        """
        Usage:
            yolo.xyscales = [1.2, 1.1, 1.05]
            yolo.xyscales = np.array([1.2, 1.1, 1.05])
            print(yolo.xyscales)
        """
        return self._xyscales

    @xyscales.setter
    def xyscales(self, xyscales: Union[list, tuple, np.ndarray]):
        if isinstance(xyscales, (list, tuple)):
            self._xyscales = np.array(xyscales)
        elif isinstance(xyscales, np.ndarray):
            self._xyscales = xyscales

    def make_model(self):
        # pylint: disable=missing-function-docstring
        self._has_weights = False
        tf.keras.backend.clear_session()
        self.model = yolov4.YOLOv4(
            anchors=self.anchors,
            input_size=self.input_size,
            num_classes=len(self.classes),
            xyscales=self.xyscales,
        )
        # [batch, height, width, channel]
        self.model(tf.keras.layers.Input([self.input_size, self.input_size, 3]))

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("yolov4.weights", weights_type="yolo")
            yolo.load_weights("checkpoints")
        """
        if weights_type == "yolo":
            weights.load_yolov4(self.model, weights_path)
        elif weights_type == "tf":
            self.model.load_weights(weights_path).expect_partial()

        self._has_weights = True

    def save_as_tflite(self, tflite_path):
        """
        Save model and weights as tflite

        Usage:
            yolo.save_as_tflite("yolov4.tflite")
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.allow_custom_ops = True
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(tflite_path, "wb") as fd:
            fd.write(tflite_model)

    def predict(self, frame: np.ndarray):
        image_data = media.resize(frame, self.input_size)
        image_data = image_data / 255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        candidates = self.model.predict(image_data)
        _candidates = []
        for candidate in candidates:
            batch_size = candidate.shape[0]
            grid_size = candidate.shape[1]
            _candidates.append(
                tf.reshape(
                    candidate, shape=(batch_size, grid_size * grid_size * 3, -1)
                )
            )
        candidates = np.concatenate(_candidates, axis=1)
        candidates = predict.reduce_bbox_candidates(candidates, self.input_size)
        candidates = predict.fit_predicted_bboxes_to_original(
            candidates, frame.shape
        )
        return candidates

    def inference(self, media_path, is_image=True, cv_waitKey_delay=10):
        if not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))
        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            prev_time = time.time()
            bboxes = self.predict(frame)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" % (1000 * exec_time)
            print(info)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = media.draw_bbox(frame, bboxes[0], self.classes)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", image)
        else:
            vid = cv2.VideoCapture(media_path)
            while True:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    break

                prev_time = time.time()
                bboxes = self.predict(frame)
                curr_time = time.time()
                exec_time = curr_time - prev_time
                info = "time: %.2f ms" % (1000 * exec_time)
                print(info)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = media.draw_bbox(frame, bboxes[0], self.classes)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", image)
                if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                    break

        print("YOLOv4: Inference is finished")
        while cv2.waitKey(10) & 0xFF != ord("q"):
            pass
        cv2.destroyWindow("result")

    def load_dataset(
        self, dataset_path, dataset_type="converted_coco", training=True
    ):
        return dataset.Dataset(
            anchors=self.anchors,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            data_augmentation=True if training else False,
            input_size=self.input_size,
            num_classes=len(self.classes),
            strides=self.strides,
            xyscales=self.xyscales,
        )

    def compile(self, iou_type: str = "giou", learning_rate: float = 1e-5):
        self.model.compile(iou_type=iou_type, learning_rate=learning_rate)

    def fit(
        self, data_set, epochs, batch_size: int = 32, subdivisions: int = 16
    ):
        """
        Train
        """
        batch = batch_size // subdivisions
        total_bboxes = 0
        for epoch in range(epochs):
            start_time = time.time()
            avg_loss = 0
            for _ in range(subdivisions):
                _batch_data = []
                for _ in range(batch):
                    _batch_data.append(next(data_set))

                batch_data = (
                    tf.concat([x[0] for x in _batch_data], axis=0),
                    (
                        tf.concat([x[1][0] for x in _batch_data], axis=0),
                        tf.concat([x[1][1] for x in _batch_data], axis=0),
                        tf.concat([x[1][2] for x in _batch_data], axis=0),
                    ),
                )

                train_data = self.model.train_step(batch_data)
                for i in range(3):
                    tf.print(
                        "count: {:2d} class_loss = {:7.2f}, xiou_loss = {:4.2f}, total_loss = {:7.2f}".format(
                            int(train_data[0][i]),
                            train_data[1][i],
                            train_data[2][i],
                            train_data[3][i],
                        )
                    )
                    total_bboxes += int(train_data[0][i])
                avg_loss += train_data[4]
                tf.print("total_bbox = {}".format(total_bboxes))

            avg_loss /= subdivisions

            tf.print(
                "\nepoch: {}, {:7.2f} avg loss, {:5.2f} sec\n".format(
                    epoch + 1, avg_loss, time.time() - start_time
                )
            )
            if epoch % 100 == 99:
                self.model.save_weights("checkpoints_{}".format(epoch + 1))
