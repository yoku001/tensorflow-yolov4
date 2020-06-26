import cv2
import random
import colorsys
import numpy as np


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names


def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_paded = image_paded / 255.0

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors,
        )
    )

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = "%s: %.2f" % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, fontScale, thickness=bbox_thick // 2
            )[0]
            cv2.rectangle(
                image,
                c1,
                (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                bbox_color,
                -1,
            )  # filled

            cv2.putText(
                image,
                bbox_mess,
                (c1[0], c1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

    return image


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
        boxes1[..., 3] - boxes1[..., 1]
    )
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
        boxes2[..., 3] - boxes2[..., 1]
    )

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def bboxes_ciou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    up = np.maximum(boxes1[..., 1], boxes2[..., 1])
    right = np.maximum(boxes1[..., 2], boxes2[..., 2])
    down = np.maximum(boxes1[..., 3], boxes2[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bboxes_iou(boxes1, boxes2)

    ax = (boxes1[..., 0] + boxes1[..., 2]) / 2
    ay = (boxes1[..., 1] + boxes1[..., 3]) / 2
    bx = (boxes2[..., 0] + boxes2[..., 2]) / 2
    by = (boxes2[..., 1] + boxes2[..., 3]) / 2

    u = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
    d = u / c

    aw = boxes1[..., 2] - boxes1[..., 0]
    ah = boxes1[..., 3] - boxes1[..., 1]
    bw = boxes2[..., 2] - boxes2[..., 0]
    bh = boxes2[..., 3] - boxes2[..., 1]

    ar_gt = bw / bh
    ar_pred = aw / ah

    ar_loss = (
        4
        / (np.pi * np.pi)
        * (np.arctan(ar_gt) - np.arctan(ar_pred))
        * (np.arctan(ar_gt) - np.arctan(ar_pred))
    )
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


def nms(bboxes, iou_threshold, sigma=0.3, method="nms"):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ["nms", "soft-nms"]

            if method == "nms":
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == "soft-nms":
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def reduce_bbox_candidates(candidates, input_size, threshold):
    """
    @param candidates: [[center_x, center_y, w, h, class_id, propability], ...]
    """

    """
    Remove low socre candidates
    This step should be the first !!
    """
    classes = np.argmax(candidates[:, 5:], axis=-1)
    scores = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), classes + 5]
    )
    candidates = candidates[scores > threshold, :]

    # Remove out of range candidates
    half = candidates[:, 2:4] * 0.5
    candidates = candidates[candidates[:, 0] - half[:, 0] >= 0, :]
    candidates = candidates[candidates[:, 0] + half[:, 0] <= 1, :]
    candidates = candidates[candidates[:, 1] - half[:, 1] >= 0, :]
    candidates = candidates[candidates[:, 1] + half[:, 1] <= 1, :]

    # Remove small candidates
    candidates = candidates[
        np.logical_and(
            candidates[:, 2] > 2 / input_size,
            candidates[:, 3] > 2 / input_size,
        ),
        :,
    ]

    classes = np.argmax(candidates[:, 5:], axis=-1)
    scores = (
        candidates[:, 4] * candidates[np.arange(len(candidates)), classes + 5]
    )

    candidates = np.concatenate(
        [candidates[:, :4], classes[:, np.newaxis], scores[:, np.newaxis],],
        axis=-1,
    )
    return candidates


def fit_predicted_bboxes_to_original(bboxes, original_shape):
    """
    @param candidates: [[center_x, center_y, w, h, class_id, propability], ...]
    """

    height = original_shape[0]
    width = original_shape[1]

    bboxes = np.copy(bboxes)
    if width > height:
        w_h = width / height
        bboxes[:, 1] = w_h * (bboxes[:, 1] - 0.5) + 0.5
        bboxes[:, 3] = w_h * bboxes[:, 3]
    elif width < height:
        h_w = height / width
        bboxes[:, 0] = h_w * (bboxes[:, 0] - 0.5) + 0.5
        bboxes[:, 2] = h_w * bboxes[:, 2]

    return bboxes


def freeze_all(model, frozen=True):
    import tensorflow as tf

    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def unfreeze_all(model, frozen=False):
    import tensorflow as tf

    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)
