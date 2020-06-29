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
import tensorflow as tf


def make_compiled_loss(
    model, iou_type: str = "giou", iou_loss_threshold: float = 0.5
):
    if iou_type == "giou":
        xiou_func = bbox_giou

    def compiled_loss(y, y_pred):
        """
        @param y:      (batch, candidates, (x, y, w, h, score, c0, c1, ...))
        @param y_pred: (batch, candidates, (x, y, w, h, score, c0, c1, ...))
        """
        y_xywh = y[..., 0:4]
        y_score = y[..., 4:5]
        y_classes = y[..., 5:]

        y_pred_xywh = y_pred[..., 0:4]
        y_pred_raw_score = y_pred[..., 4:5]
        y_pred_raw_classes = y_pred[..., 5:]

        y_pred_score = tf.keras.activations.sigmoid(y_pred_raw_score)

        # XIoU loss
        xiou = tf.expand_dims(xiou_func(y_xywh, y_pred_xywh), axis=-1)
        xiou_loss = (
            y_score * (2.0 - y_xywh[..., 2:3] * y_xywh[..., 3:4]) * (1 - xiou)
        )

        # Score loss
        """
        @param bboxes1: batch, candidates, 1,       xywh
        @param bboxes2: batch, 1         , answers, xywh
        @return batch, candidates, answers
        """
        max_iou = []
        for i in range(len(y_score)):  # batch
            mask = y_score[i, ..., 0] > 0.5
            iou = bbox_iou(
                tf.expand_dims(y_pred_xywh[i, ...], axis=-2),
                tf.expand_dims(tf.boolean_mask(y_xywh[i, ...], mask), axis=-3),
            )
            max_iou.append(
                tf.reshape(tf.reduce_max(iou, axis=-1), shape=(1, -1, 1))
            )
        max_iou = tf.concat(max_iou, axis=0)

        score_loss = (
            tf.pow(y_score - y_pred_score, 2)
            * (
                y_score
                + (1.0 - y_score)
                * tf.cast(max_iou < iou_loss_threshold, tf.float32)
            )
            * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_score, logits=y_pred_raw_score
            )
        )
        # Classes loss
        classes_loss = y_score * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_classes, logits=y_pred_raw_classes
        )

        xiou_loss = tf.reduce_mean(tf.reduce_sum(xiou_loss, axis=[1, 2]))
        score_loss = tf.reduce_mean(tf.reduce_sum(score_loss, axis=[1, 2]))
        classes_loss = tf.reduce_mean(tf.reduce_sum(classes_loss, axis=[1, 2]))

        return xiou_loss, score_loss, classes_loss

    return compiled_loss


def bbox_iou(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = tf.maximum(bboxes1_area + bboxes2_area - inter_area, 0.000001)

    iou = inter_area / union_area

    return iou


def bbox_giou(bboxes1, bboxes2):
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1 = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2 = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])
    right_down = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = tf.maximum(bboxes1_area + bboxes2_area - inter_area, 0.000001)

    iou = inter_area / union_area

    enclose_left_up = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
    enclose_right_down = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = tf.maximum(
        enclose_section[..., 0] * enclose_section[..., 1], 0.000001
    )

    giou = iou - (enclose_area - union_area) / enclose_area

    return giou


def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat(
        [
            boxes1[..., :2] - boxes1[..., 2:] * 0.5,
            boxes1[..., :2] + boxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    boxes2_coor = tf.concat(
        [
            boxes2[..., :2] - boxes2[..., 2:] * 0.5,
            boxes2[..., :2] + boxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (
        boxes1[..., 0] - boxes2[..., 0]
    ) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = (
        4
        / (np.pi * np.pi)
        * (tf.atan(ar_gt) - tf.atan(ar_pred))
        * (tf.atan(ar_gt) - tf.atan(ar_pred))
    )
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term
