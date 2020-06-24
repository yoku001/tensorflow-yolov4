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

import cv2
import numpy as np
import os
import shutil
import tensorflow as tf
import time
from typing import Union

from ..core import dataset
from ..core import utils
from ..core import yolov4


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
            self._classes = utils.read_class_names(data)
        elif isinstance(data, dict):
            self._classes = data
        else:
            raise TypeError("YoloV4: Set classes path or dictionary")

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
        self._has_weights = False
        tf.keras.backend.clear_session()
        self.model = yolov4.YOLOv4(num_classes=len(self.classes))
        # [batch, height, width, channel]
        self.model(tf.keras.layers.Input([self.input_size, self.input_size, 3]))

    def load_weights(self, path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("yolov4.weights", weights_type="yolo")
            yolo.load_weights("checkpoints")
        """
        if weights_type == "yolo":
            utils.load_weights(self.model, path)
        elif weights_type == "tf":
            self.model.load_weights(path).expect_partial()

        self._has_weights = True

    def predict(self, frame):
        frame_size = frame.shape[:2]

        image_data = utils.image_preprocess(
            np.copy(frame), [self.input_size, self.input_size]
        )
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)

        pred_bbox = utils.postprocess_bbbox(
            pred_bbox, self.anchors, self.strides, self.xyscales
        )
        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, self.input_size, 0.25
        )
        bboxes = utils.nms(bboxes, 0.213, method="nms")

        return utils.draw_bbox(frame, bboxes, self.classes)

    def inference(self, media_path, is_image=True, cv_waitKey_delay=10):
        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = self.predict(frame)

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            cv2.waitKey(cv_waitKey_delay)
        else:
            vid = cv2.VideoCapture(media_path)
            while True:
                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("No image! Try with another video format")

                prev_time = time.time()
                image = self.predict(frame)
                curr_time = time.time()
                exec_time = curr_time - prev_time

                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                print(info)
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("result", result)
                if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                    break

    def train(
        self,
        train_annote_path: str,
        test_annote_path: str,
        dataset_type: str = "converted_coco",
        epochs: int = 50,
        iou_loss_threshold: float = 0.5,
        learning_rate_init: float = 1e-3,
        learning_rate_end: float = 1e-6,
        log_dir_path: str = "./log",
        save_interval: int = 1,
        trained_weights_path: str = "./checkpoints",
    ):

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(e)

        trainset = dataset.Dataset(
            annot_path=train_annote_path,
            classes=self.classes,
            anchors=self.anchors,
            input_sizes=self.input_size,
            dataset_type=dataset_type,
        )
        testset = dataset.Dataset(
            annot_path=test_annote_path,
            classes=self.classes,
            anchors=self.anchors,
            input_sizes=self.input_size,
            is_training=False,
            dataset_type=dataset_type,
        )

        isfreeze = False

        if self._has_weights:
            first_stage_epochs = int(epochs * 0.3)
        else:
            first_stage_epochs = 0

        steps_per_epoch = len(trainset)
        second_stage_epochs = epochs - first_stage_epochs
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = 2 * steps_per_epoch
        total_steps = (
            first_stage_epochs + second_stage_epochs
        ) * steps_per_epoch

        optimizer = tf.keras.optimizers.Adam()
        if os.path.exists(log_dir_path):
            shutil.rmtree(log_dir_path)
        writer = tf.summary.create_file_writer(log_dir_path)

        def decode_train(bboxes, index: int):
            conv_shape = tf.shape(bboxes)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]

            (dxdy, wh, raw_score, raw_classes) = tf.split(
                bboxes, (2, 2, 1, len(self.classes)), axis=-1
            )

            grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
            grid = np.expand_dims(np.stack(grid, axis=-1), axis=2)
            """
            grid(i, j, 1, 2) => grid top left coordinates
            [
                [ [[0, 0]], [[1, 0]], [[2, 0]], ...],
                [ [[0, 1]], [[1, 1]], [[2, 1]], ...],
            ]
            """

            # grid(1, i, j, 3, 2)
            grid = np.tile(np.expand_dims(grid, axis=0), [1, 1, 1, 3, 1])
            grid = grid.astype(np.float)

            pred_xy = (
                ((dxdy - 0.5) * self.xyscales[index]) + 0.5 + grid
            ) * self.strides[index]
            pred_wh = tf.exp(wh) * self.anchors[index]
            pred_score = tf.sigmoid(raw_score)
            pred_classes = tf.sigmoid(raw_classes)

            return tf.concat(
                [pred_xy, pred_wh, pred_score, pred_classes], axis=-1
            )

        def train_step(image_data, target):
            with tf.GradientTape() as tape:
                bboxes = self.model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(3):
                    conv, pred = bboxes[i], decode_train(bboxes[i], i)
                    loss_items = yolov4.compute_loss(
                        pred,
                        conv,
                        target[i][0],
                        target[i][1],
                        strides=self.strides,
                        num_class=len(self.classes),
                        iou_loss_threshold=iou_loss_threshold,
                        i=i,
                    )
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(
                    total_loss, self.model.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
                tf.print(
                    "=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f"
                    % (
                        global_steps,
                        optimizer.lr.numpy(),
                        giou_loss,
                        conf_loss,
                        prob_loss,
                        total_loss,
                    )
                )
                # update learning rate
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * learning_rate_init
                else:
                    lr = learning_rate_end + 0.5 * (
                        learning_rate_init - learning_rate_end
                    ) * (
                        (
                            1
                            + tf.cos(
                                (global_steps - warmup_steps)
                                / (total_steps - warmup_steps)
                                * np.pi
                            )
                        )
                    )
                optimizer.lr.assign(lr.numpy())

                # writing summary data
                writer.as_default()
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar(
                    "loss/total_loss", total_loss, step=global_steps
                )
                tf.summary.scalar(
                    "loss/giou_loss", giou_loss, step=global_steps
                )
                tf.summary.scalar(
                    "loss/conf_loss", conf_loss, step=global_steps
                )
                tf.summary.scalar(
                    "loss/prob_loss", prob_loss, step=global_steps
                )
                writer.flush()

        def test_step(image_data, target):
            with tf.GradientTape() as tape:
                bboxes = self.model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(3):
                    conv, pred = bboxes[i], decode_train(bboxes[i], i)
                    loss_items = yolov4.compute_loss(
                        pred,
                        conv,
                        target[i][0],
                        target[i][1],
                        strides=self.strides,
                        num_class=len(self.classes),
                        iou_loss_threshold=iou_loss_threshold,
                        i=i,
                    )
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                tf.print(
                    "=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f"
                    % (
                        global_steps,
                        giou_loss,
                        conf_loss,
                        prob_loss,
                        total_loss,
                    )
                )

        for epoch in range(epochs):
            if epoch < first_stage_epochs:
                if not isfreeze:
                    isfreeze = True
                    for name in [
                        "yolo_conv2d_93",
                        "yolo_conv2d_101",
                        "yolo_conv2d_109",
                    ]:
                        freeze = self.model.get_layer(name)
                        utils.freeze_all(freeze)
            elif epoch >= first_stage_epochs:
                if isfreeze:
                    isfreeze = False
                    for name in [
                        "yolo_conv2d_93",
                        "yolo_conv2d_101",
                        "yolo_conv2d_109",
                    ]:
                        freeze = self.model.get_layer(name)
                        utils.unfreeze_all(freeze)

            for image_data, target in trainset:
                train_step(image_data, target)
            for image_data, target in testset:
                test_step(image_data, target)

            if epoch % save_interval == 0:
                self.model.save_weights(trained_weights_path)

        self.model.save_weights(trained_weights_path)
