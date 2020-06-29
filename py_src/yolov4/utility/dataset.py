"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
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

import os
import cv2
import numpy as np

from . import media
from . import train


class Dataset(object):
    def __init__(
        self,
        anchors: np.ndarray = None,
        datasets_path: str = None,
        datasets_type: str = "converted_coco",
        input_size: int = 416,
        num_classes: int = None,
        strides: np.ndarray = None,
        training: bool = True,
    ):
        self.anchors_ratio = anchors.reshape(-1, 2) / input_size
        self.datasets_path = datasets_path
        self.datasets_type = datasets_type
        self.grid_size = input_size // strides
        self.input_size = input_size
        self.num_candidates = np.sum(np.power(self.grid_size, 2)) * len(anchors)
        self.num_classes = num_classes
        self.training = training

        self.datasets = self.load_dataset()

        self.count = 0
        np.random.shuffle(self.datasets)

    def load_dataset(self):
        """
        @return
            yolo: [[image_path, [[x, y, w, h, class_id], ...]], ...]
            converted_coco: unit=> pixel
                [[image_path, [[x, y, w, h, class_id], ...]], ...]
        """
        datasets = []

        with open(self.datasets_path, "r") as fd:
            txt = fd.readlines()
            if self.datasets_type == "converted_coco":
                for line in txt:
                    # line: "<image_path> xmin,ymin,xmax,ymax,class_id ..."
                    bboxes = line.strip().split()
                    image_path = bboxes[0]
                    xywhc_s = np.zeros((len(bboxes) - 1, 5))
                    for i, bbox in enumerate(bboxes[1:]):
                        # bbox = "xmin,ymin,xmax,ymax,class_id"
                        bbox = list(map(int, bbox.split(",")))
                        xywhc_s[i, :] = (
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            bbox[4],
                        )
                    datasets.append([image_path, xywhc_s])

            elif self.datasets_type == "yolo":
                for line in txt:
                    # line: "<image_path>"
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd2:
                        bboxes = fd2.readlines()
                        xywhc_s = np.zeros((len(bboxes), 5))
                        for i, bbox in enumerate(bboxes):
                            # bbox = class_id, x, y, w, h
                            bbox = bbox.strip()
                            bbox = list(map(float, bbox.split(",")))
                            xywhc_s[i, :] = (
                                *bbox[1:],
                                bbox[0],
                            )
                        datasets.append([image_path, xywhc_s])
        return datasets

    def bboxes_to_ground_truth(self, bboxes):
        """
        @param bboxes: [[x, y, w, h, class_id], ...]

        @return [[x, y, w, h, score, c0, c1, ...], ...]
        """
        ground_truth = np.zeros(
            (self.num_candidates, 5 + self.num_classes), dtype=np.float32
        )

        for bbox in bboxes:
            # [x, y, w, h, class_id]
            xywh = np.array(bbox[:4], dtype=np.float32)
            class_id = int(bbox[4])

            # smooth_onehot = [0.xx, ... , 1-(0.xx*(n-1)), 0.xx, ...]
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[class_id] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes, dtype=np.float32
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            scaler = np.tile(self.grid_size.reshape(-1, 1), (1, 6)).reshape(
                -1, 2
            )
            grid = np.floor(xywh[0:2] * scaler, dtype=np.float32)
            anchors_xywh = np.zeros((9, 4), dtype=np.float32)
            anchors_xywh[:, 0:2] = (grid + 0.5) / scaler
            anchors_xywh[:, 2:4] = self.anchors_ratio

            iou = train.bbox_iou(xywh, anchors_xywh)

            iou_mask = iou > 0.3

            if np.any(iou_mask):
                for i, mask in enumerate(iou_mask):
                    if mask:
                        """
                a: anchor
                s_a_0, s_a_1, s_a_2, m_a_0, m_a_1, m_a_2, l_a_0, l_a_1, l_a_2

                s_grid, s_grid, s_anchors, (x,y,w,h,score,classes)
                m_grid, m_grid, m_anchors, (x,y,w,h,score,classes)
                l_grid, l_grid, l_anchors, (x,y,w,h,score,classes)
                => grid*grid*anchors, (x,y,w,h,score,classes)
                        """
                        index = 0
                        size_index = i // 3
                        anchor_index = i % 3
                        for j in range(size_index):
                            index += self.grid_size[j] * self.grid_size[j] * 3
                        index += grid[i][1] * self.grid_size[size_index] * 3
                        index += grid[i][0] * 3
                        index += anchor_index
                        index = int(index)

                        ground_truth[index, 0:4] = xywh
                        ground_truth[index, 4:5] = 1.0
                        ground_truth[index, 5:] = smooth_onehot
            else:
                i = np.argmax(iou)
                index = 0
                size_index = i // 3
                anchor_index = i % 3
                for j in range(size_index):
                    index += self.grid_size[j] * self.grid_size[j] * 3
                index += grid[i][1] * self.grid_size[size_index] * 3
                index += grid[i][0] * 3
                index += anchor_index
                index = int(index)

                ground_truth[index, 0:4] = xywh
                ground_truth[index, 4:5] = 1.0
                ground_truth[index, 5:] = smooth_onehot

        return ground_truth

    def preprocess_dataset(self, dataset):
        """
        @param dataset:
            yolo: [image_path, [[x, y, w, h, class_id], ...]]
            converted_coco: unit=> pixel
                [image_path, [[x, y, w, h, class_id], ...]]

        @return image / 255, ground_truth
        """
        image_path = dataset[0]
        if not os.path.exists(image_path):
            raise KeyError("{} does not exist".format(image_path))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.datasets_type == "converted_coco":
            height, width, _ = image.shape
            dataset[1] = dataset[1] / np.array(
                [width, height, width, height, 1]
            )

        resized_image, resized_bboxes = media.resize(
            image, self.input_size, dataset[1]
        )
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)
        ground_truth = self.bboxes_to_ground_truth(resized_bboxes)
        ground_truth = np.expand_dims(ground_truth, axis=0)

        if self.training:
            # TODO
            # BoF functions
            pass
        return resized_image, ground_truth

    def __iter__(self):
        self.count = 0
        np.random.shuffle(self.datasets)
        return self

    def __next__(self):
        x, y = self.preprocess_dataset(self.datasets[self.count])

        self.count += 1
        if self.count == len(self.datasets):
            np.random.shuffle(self.datasets)
            self.count = 0

        return x, y

    def __len__(self):
        return len(self.datasets)
