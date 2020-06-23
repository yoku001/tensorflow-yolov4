import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from . import utils
from . import common
from .common import YOLOConv2D
from .backbone import CSPDarknet53


class YOLOv4(Model):
    def __init__(
        self,
        anchors: np.ndarray,
        num_classes: int,
        strides: np.ndarray,
        xyscale: np.ndarray,
        **k
    ):
        super(YOLOv4, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.strides = strides
        self.xyscale = xyscale

        self.csp_darknet53 = CSPDarknet53()

        self.conv78 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.upSampling78 = layers.UpSampling2D()
        self.conv79 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.concat78_79 = layers.Concatenate(axis=-1)

        self.conv80 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv81 = YOLOConv2D(
            filters=512, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv82 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv83 = YOLOConv2D(
            filters=512, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv84 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )

        self.conv85 = YOLOConv2D(
            filters=128, kernel_size=1, strides=1, activation="leaky"
        )
        self.upSampling85 = layers.UpSampling2D()
        self.conv86 = YOLOConv2D(
            filters=128, kernel_size=1, strides=1, activation="leaky"
        )
        self.concat85_86 = layers.Concatenate(axis=-1)

        self.conv87 = YOLOConv2D(
            filters=128, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv88 = YOLOConv2D(
            filters=256, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv89 = YOLOConv2D(
            filters=128, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv90 = YOLOConv2D(
            filters=256, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv91 = YOLOConv2D(
            filters=128, kernel_size=1, strides=1, activation="leaky"
        )

        self.conv92 = YOLOConv2D(
            filters=256, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv93 = YOLOConv2D(
            filters=3 * (self.num_classes + 5),
            kernel_size=1,
            strides=1,
            activation=None,
        )

        self.conv94 = YOLOConv2D(
            filters=256, kernel_size=3, strides=2, activation="leaky"
        )
        self.concat84_94 = layers.Concatenate(axis=-1)

        self.conv95 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv96 = YOLOConv2D(
            filters=512, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv97 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv98 = YOLOConv2D(
            filters=512, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv99 = YOLOConv2D(
            filters=256, kernel_size=1, strides=1, activation="leaky"
        )

        self.conv100 = YOLOConv2D(
            filters=512, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv101 = YOLOConv2D(
            filters=3 * (self.num_classes + 5),
            kernel_size=1,
            strides=1,
            activation=None,
        )

        self.conv102 = YOLOConv2D(
            filters=512, kernel_size=3, strides=2, activation="leaky"
        )
        self.concat77_102 = layers.Concatenate(axis=-1)

        self.conv103 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv104 = YOLOConv2D(
            filters=1024, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv105 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )
        self.conv106 = YOLOConv2D(
            filters=1024, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv107 = YOLOConv2D(
            filters=512, kernel_size=1, strides=1, activation="leaky"
        )

        self.conv108 = YOLOConv2D(
            filters=1024, kernel_size=3, strides=1, activation="leaky"
        )
        self.conv109 = YOLOConv2D(
            filters=3 * (self.num_classes + 5),
            kernel_size=1,
            strides=1,
            activation=None,
        )

    def call(self, x, training: bool = False):
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

        sbbox = self.conv92(x2)
        sbbox = self.conv93(sbbox)

        x2 = self.conv94(x2)
        x2 = self.concat84_94([x2, x1])

        x2 = self.conv95(x2)
        x2 = self.conv96(x2)
        x2 = self.conv97(x2)
        x2 = self.conv98(x2)
        x2 = self.conv99(x2)

        mbbox = self.conv100(x2)
        mbbox = self.conv101(mbbox)

        x2 = self.conv102(x2)
        x2 = self.concat77_102([x2, route3])

        x2 = self.conv103(x2)
        x2 = self.conv104(x2)
        x2 = self.conv105(x2)
        x2 = self.conv106(x2)
        x2 = self.conv107(x2)

        lbbox = self.conv108(x2)
        lbbox = self.conv109(lbbox)

        bboxes = []
        for i, fm in enumerate([sbbox, mbbox, lbbox]):
            if training:
                bbox = decode_train(
                    fm,
                    self.num_classes,
                    self.strides,
                    self.anchors,
                    i,
                    self.xyscale,
                )
                bboxes.append(fm)

            else:
                bbox = decode(fm, self.num_classes, i)

            bboxes.append(bbox)

        return bboxes


def decode(conv_output, num_class, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + num_class)
    )
    conv_raw_xywh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (4, 1, num_class), axis=-1
    )

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([conv_raw_xywh, pred_conf, pred_prob], axis=-1)


def decode_train(
    conv_output, num_class, strides, anchors, i=0, xyscale=[1, 1, 1]
):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + num_class)
    )
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
        conv_output, (2, 2, 1, num_class), axis=-1
    )

    x = tf.tile(
        tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0),
        [output_size, 1],
    )
    y = tf.tile(
        tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1),
        [1, output_size],
    )
    xy_grid = tf.expand_dims(
        tf.stack([x, y], axis=-1), axis=2
    )  # [gx, gy, 1, 2]
    # xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
    # xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (
        (tf.sigmoid(conv_raw_dxdy) * xyscale[i])
        - 0.5 * (xyscale[i] - 1)
        + xy_grid
    ) * strides[i]
    pred_wh = tf.exp(conv_raw_dwdh) * anchors[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

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

    left_up = tf.maximum(boxes1_coor[..., :2], boxes1_coor[..., :2])
    right_down = tf.minimum(boxes2_coor[..., 2:], boxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


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


def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat(
        [
            boxes1[..., :2] - boxes1[..., 2:] * 0.5,
            boxes1[..., :2] + boxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    boxes2 = tf.concat(
        [
            boxes2[..., :2] - boxes2[..., 2:] * 0.5,
            boxes2[..., :2] + boxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    boxes1 = tf.concat(
        [
            tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
            tf.maximum(boxes1[..., :2], boxes1[..., 2:]),
        ],
        axis=-1,
    )
    boxes2 = tf.concat(
        [
            tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
            tf.maximum(boxes2[..., :2], boxes2[..., 2:]),
        ],
        axis=-1,
    )

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
        boxes1[..., 3] - boxes1[..., 1]
    )
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
        boxes2[..., 3] - boxes2[..., 1]
    )

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(
    pred, conv, label, bboxes, strides, num_class, iou_loss_threshold, i=0
):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = strides[i] * output_size
    conv = tf.reshape(
        conv, (batch_size, output_size, output_size, 3, 5 + num_class)
    )

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[
        :, :, :, :, 3:4
    ] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(
        pred_xywh[:, :, :, :, np.newaxis, :],
        bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :],
    )
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(
        max_iou < iou_loss_threshold, tf.float32
    )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
        respond_bbox
        * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf
        )
        + respond_bgd
        * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=respond_bbox, logits=conv_raw_conf
        )
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_prob, logits=conv_raw_prob
    )

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
