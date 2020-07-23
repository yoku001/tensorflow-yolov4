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

try:
    import tensorflow.lite as tflite
except ModuleNotFoundError:
    import tflite_runtime.interpreter as tflite

from ..common import media, predict
from ..common.base_class import BaseClass


class YOLOv4(BaseClass):
    def __init__(self, tiny: bool = False, tpu: bool = False):
        """
        Default configuration
        """
        super(YOLOv4, self).__init__(tiny=tiny, tpu=tpu)
        self.grid_coord = []
        self.input_index = None
        self.interpreter = None
        self.output_index = None
        self.output_size = None

    def load_tflite(self, tflite_path):
        if self.tpu:
            self.interpreter = tflite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[
                    tflite.load_delegate("libedgetpu.so.1")
                ],
            )
        else:
            self.interpreter = tflite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]
        self.input_size = input_details["shape"][1]
        self.input_index = input_details["index"]
        output_details = self.interpreter.get_output_details()
        self.output_index = [details["index"] for details in output_details]
        if self.tpu:
            # sig, raw, sig, raw, ...
            self.output_size = [
                output_details[2 * i]["shape"][1]
                for i in range(len(output_details) // 2)
            ]
            self.grid_coord = []
            for _size in self.output_size:
                xy_grid = np.meshgrid(np.arange(_size), np.arange(_size))
                xy_grid = np.stack(xy_grid, axis=-1)
                xy_grid = xy_grid[np.newaxis, ...]
                self.grid_coord.append(xy_grid)

    #############
    # Inference #
    #############

    def tpu_hair(self, x):
        x = [np.split(_x, 3, axis=-1) for _x in x]

        for i in range(len(self.output_size)):
            for j in range(3):
                # pylint: disable=unbalanced-tuple-unpacking
                # sig
                txty, _, conf_prob = np.split(x[2 * i][j], (2, 4), axis=-1)
                # raw
                _, twth, _ = np.split(x[2 * i + 1][j], (2, 4), axis=-1)
                txty = (txty - 0.5) * self.xyscales[i] + 0.5
                bxby = (txty + self.grid_coord[i]) / self.output_size[i]
                bwbh = (self.anchors[i][j] / self.input_size) * np.exp(twth)
                x[2 * i][j] = np.concatenate([bxby, bwbh, conf_prob], axis=-1)

        pred = [
            np.concatenate(x[2 * i], axis=-1)
            for i in range(len(self.output_size))
        ]
        return pred

    def predict(self, frame: np.ndarray):
        """
        Predict one frame

        @param frame: Dim(height, width, channels)

        @return pred_bboxes == Dim(-1, (x, y, w, h, class_id, probability))
        """
        # image_data == Dim(1, input_szie, input_size, channels)
        image_data = self.resize_image(frame)
        image_data = image_data / 255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # s_pred, m_pred, l_pred
        # x_pred == Dim(1, output_size, output_size, anchors, (bbox))
        self.interpreter.set_tensor(self.input_index, image_data)
        self.interpreter.invoke()
        candidates = [
            self.interpreter.get_tensor(index) for index in self.output_index
        ]
        if self.tpu:
            candidates = self.tpu_hair(candidates)
        _candidates = []
        for candidate in candidates:
            grid_size = candidate.shape[1]
            _candidates.append(
                np.reshape(candidate[0], (1, grid_size * grid_size * 3, -1))
            )
        candidates = np.concatenate(_candidates, axis=1)

        pred_bboxes = self.candidates_to_pred_bboxes(candidates[0])
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes

    def inference(self, media_path, is_image=True, cv_waitKey_delay=10):
        if not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))
        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            bboxes = self.predict(frame)
            exec_time = time.time() - start_time
            print("time: {:.2f} ms".format(exec_time * 1000))

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = self.draw_bboxes(frame, bboxes)
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

                start_time = time.time()
                bboxes = self.predict(frame)
                curr_time = time.time()
                exec_time = curr_time - start_time
                info = "time: %.2f ms" % (1000 * exec_time)
                print(info)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = self.draw_bboxes(frame, bboxes)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", image)
                if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                    break

        print("YOLOv4: Inference is finished")
        while cv2.waitKey(10) & 0xFF != ord("q"):
            pass
        cv2.destroyWindow("result")
