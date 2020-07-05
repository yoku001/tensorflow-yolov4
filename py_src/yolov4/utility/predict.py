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

import numpy as np


def DIoU_NMS(candidates, threshold):
    """
    Distance Intersection over Union(DIoU)
    Non-Maximum Suppression(NMS)

    @param candidates: [[center_x, center_y, w, h, class_id, propability], ...]
    """
    bboxes = []
    for class_id in set(candidates[:, 4]):
        class_bboxes = candidates[candidates[:, 4] == class_id]
        if class_bboxes.shape[0] == 1:
            # One candidate
            bboxes.append(class_bboxes)
            continue

        while True:
            half = class_bboxes[:, 2:4] * 0.5
            M_index = np.argmax(class_bboxes[:, 5])
            M_bbox = class_bboxes[M_index, :]
            M_half = half[M_index, :]
            # Max probability
            bboxes.append(M_bbox[np.newaxis, :])

            enclose_left = np.minimum(
                class_bboxes[:, 0] - half[:, 0], M_bbox[0] - M_half[0],
            )
            enclose_right = np.maximum(
                class_bboxes[:, 0] + half[:, 0], M_bbox[0] + M_half[0],
            )
            enclose_top = np.minimum(
                class_bboxes[:, 1] - half[:, 1], M_bbox[1] - M_half[1],
            )
            enclose_bottom = np.maximum(
                class_bboxes[:, 1] + half[:, 1], M_bbox[1] + M_half[1],
            )

            enclose_width = enclose_right - enclose_left
            enclose_height = enclose_bottom - enclose_top

            width_mask = enclose_width >= class_bboxes[:, 2] + M_bbox[2]
            height_mask = enclose_height >= class_bboxes[:, 3] + M_bbox[3]
            other_mask = np.logical_or(width_mask, height_mask)
            other_bboxes = class_bboxes[other_mask]

            mask = np.logical_not(other_mask)
            class_bboxes = class_bboxes[mask]
            if class_bboxes.shape[0] == 1:
                if other_bboxes.shape[0] == 1:
                    bboxes.append(other_bboxes)
                    break
                else:
                    class_bboxes = other_bboxes
                    continue

            half = half[mask]
            enclose_left = enclose_left[mask]
            enclose_right = enclose_right[mask]
            enclose_top = enclose_top[mask]
            enclose_bottom = enclose_bottom[mask]

            inter_left = np.maximum(
                class_bboxes[:, 0] - half[:, 0], M_bbox[0] - M_half[0],
            )
            inter_right = np.minimum(
                class_bboxes[:, 0] + half[:, 0], M_bbox[0] + M_half[0],
            )
            inter_top = np.maximum(
                class_bboxes[:, 1] - half[:, 1], M_bbox[1] - M_half[1],
            )
            inter_bottom = np.minimum(
                class_bboxes[:, 1] + half[:, 1], M_bbox[1] + M_half[1],
            )

            class_area = class_bboxes[:, 2] * class_bboxes[:, 3]
            M_area = M_bbox[2] * M_bbox[3]
            inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
            iou = inter_area / (class_area + M_area)

            c = (enclose_right - enclose_left) * (
                enclose_right - enclose_left
            ) + (enclose_bottom - enclose_top) * (enclose_bottom - enclose_top)
            d = (class_bboxes[:, 0] - M_bbox[0]) * (
                class_bboxes[:, 0] - M_bbox[0]
            ) + (class_bboxes[:, 1] - M_bbox[1]) * (
                class_bboxes[:, 1] - M_bbox[1]
            )

            # DIoU = IoU - d^2 / c^2
            other_mask = iou - d / c < threshold
            other2_bboxes = class_bboxes[other_mask]
            if other_bboxes.shape[0] != 0 and other2_bboxes.shape[0] != 0:
                class_bboxes = np.concatenate(
                    [other_bboxes, other2_bboxes], axis=0
                )
                continue
            elif other_bboxes.shape[0] != 0:
                if other_bboxes.shape[0] == 1:
                    bboxes.append(other_bboxes)
                    break
                else:
                    class_bboxes = other_bboxes
                    continue
            elif other2_bboxes.shape[0] != 0:
                if other2_bboxes.shape[0] == 1:
                    bboxes.append(other2_bboxes)
                    break
                else:
                    class_bboxes = other2_bboxes
                    continue
            else:
                break

    if len(bboxes) == 0:
        return np.zeros(shape=(1, 6))

    return np.concatenate(bboxes, axis=0)


def reduce_bbox_candidates(
    candidates,
    input_size,
    score_threshold: float = 0.25,
    DIoU_threshold: float = 0.3,
):
    """
    @param candidates: Dim(batch, -1, (x, y, w, h, score, classes))

    @return Dim(batch, -1, (x, y, w, h, class_id, probability))
    """
    _candidates = []
    for candidate in candidates:
        # Remove low socre candidates
        # This step should be the first !!
        class_ids = np.argmax(candidate[:, 5:], axis=-1)
        scores = (
            candidate[:, 4]
            * candidate[np.arange(len(candidate)), class_ids + 5]
        )
        candidate = candidate[scores > score_threshold, :]

        # Remove out of range candidates
        half = candidate[:, 2:4] * 0.5
        mask = candidate[:, 0] - half[:, 0] >= 0
        candidate = candidate[mask, :]
        half = half[mask, :]
        mask = candidate[:, 0] + half[:, 0] <= 1
        candidate = candidate[mask, :]
        half = half[mask, :]
        mask = candidate[:, 1] - half[:, 1] >= 0
        candidate = candidate[mask, :]
        half = half[mask, :]
        mask = candidate[:, 1] + half[:, 1] <= 1
        candidate = candidate[mask, :]

        # Remove small candidates
        candidate = candidate[
            np.logical_and(
                candidate[:, 2] > 2 / input_size,
                candidate[:, 3] > 2 / input_size,
            ),
            :,
        ]

        class_ids = np.argmax(candidate[:, 5:], axis=-1)
        scores = (
            candidate[:, 4]
            * candidate[np.arange(len(candidate)), class_ids + 5]
        )

        candidate = np.concatenate(
            [
                candidate[:, :4],
                class_ids[:, np.newaxis],
                scores[:, np.newaxis],
            ],
            axis=-1,
        )

        candidate = DIoU_NMS(candidate, DIoU_threshold)

        _candidates.append(candidate)

    return _candidates


def fit_predicted_bboxes_to_original(bboxes, original_shape):
    """
    @param candidates: Dim(batch, -1, (x, y, w, h, class_id, probability))
    """

    height = original_shape[0]
    width = original_shape[1]

    _bboxes = []
    for bbox in bboxes:
        bbox = np.copy(bbox)
        if width > height:
            w_h = width / height
            bbox[:, 1] = w_h * (bbox[:, 1] - 0.5) + 0.5
            bbox[:, 3] = w_h * bbox[:, 3]
        elif width < height:
            h_w = height / width
            bbox[:, 0] = h_w * (bbox[:, 0] - 0.5) + 0.5
            bbox[:, 2] = h_w * bbox[:, 2]
        _bboxes.append(bbox)

    return _bboxes
