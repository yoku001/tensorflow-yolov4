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

    return np.concatenate(bboxes, axis=0)


def reduce_bbox_candidates(
    candidates, input_size, score_threshold, DIoU_threshold
):
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
    candidates = candidates[scores > score_threshold, :]

    # Remove out of range candidates
    half = candidates[:, 2:4] * 0.5
    mask = candidates[:, 0] - half[:, 0] >= 0
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 0] + half[:, 0] <= 1
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 1] - half[:, 1] >= 0
    candidates = candidates[mask, :]
    half = half[mask, :]
    mask = candidates[:, 1] + half[:, 1] <= 1
    candidates = candidates[mask, :]

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

    candidates = DIoU_NMS(candidates, DIoU_threshold)

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
