import tensorflow as tf


def loss_fn(labels, p):
    try:
        # iou = get_iou_grid(p, labels)

        # object_mask = labels[..., 4] > 0
        # iou_filtered = iou[tf.expand_dims(tf.expand_dims(object_mask, -1), -1)]
        #
        # p_filtered = p[object_mask]
        # labels_filtered = labels[object_mask]

        x_losses = tf.keras.losses.MeanSquaredError()(labels[..., 0], p[..., 0])
        y_losses = tf.keras.losses.MeanSquaredError()(labels[..., 1], p[..., 1])

        pos_losses = x_losses + y_losses

        h_loss = tf.keras.losses.MeanSquaredError()(labels[..., 2], p[..., 2])
        w_loss = tf.keras.losses.MeanSquaredError()(labels[..., 3], p[..., 3])
        size_losses = h_loss + w_loss

        # confidence_object_loss = tf.keras.losses.BinaryCrossentropy()(labels_filtered[..., 4],
        #                                                               iou_filtered)
        #
        # no_object_mask = labels[..., 4] == 0
        # p_noobj = p[no_object_mask]
        # labels_noobj = labels[no_object_mask]
        # confidence_no_object_loss = tf.keras.losses.BinaryCrossentropy()(labels_noobj[..., 4],
        #                                                                  p_noobj[..., 4])

        total_loss = pos_losses * 10 + size_losses
        return total_loss, pos_losses, size_losses
    except Exception as e:

        return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)


def get_iou_grid(boxes1_b, boxes2_b):
    """
    Vectorized IoU computation between two sets of bounding boxes
    in (cx, cy, w, h) format.

    Args:
        boxes1_b: tf.Tensor (..., 4)
        boxes2_b: tf.Tensor (..., 4)
    Returns:
        IoU tensor (..., num_boxes1, num_boxes2)
    """
    boxes1_b = tf.expand_dims(boxes1_b, -2)
    boxes2_b = tf.expand_dims(boxes2_b, -2)

    boxes1_b = boxes1_b[..., :4]
    boxes2_b = boxes2_b[..., :4]

    # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes1_xy = tf.concat([
        boxes1_b[..., :2] - boxes1_b[..., 2:] / 2,
        boxes1_b[..., :2] + boxes1_b[..., 2:] / 2
    ], axis=-1)

    boxes2_xy = tf.concat([
        boxes2_b[..., :2] - boxes2_b[..., 2:] / 2,
        boxes2_b[..., :2] + boxes2_b[..., 2:] / 2
    ], axis=-1)

    # Compute areas
    area1 = (boxes1_xy[..., 2] - boxes1_xy[..., 0]) * (boxes1_xy[..., 3] - boxes1_xy[..., 1])
    area2 = (boxes2_xy[..., 2] - boxes2_xy[..., 0]) * (boxes2_xy[..., 3] - boxes2_xy[..., 1])

    # Add broadcast dimensions
    boxes1_exp = tf.expand_dims(boxes1_xy, -2)  # (..., N, 1, 4)
    boxes2_exp = tf.expand_dims(boxes2_xy, -3)  # (..., 1, M, 4)

    # Intersection
    inter_ul = tf.maximum(boxes1_exp[..., :2], boxes2_exp[..., :2])
    inter_br = tf.minimum(boxes1_exp[..., 2:], boxes2_exp[..., 2:])
    inter_wh = tf.maximum(inter_br - inter_ul, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Union
    union_area = tf.expand_dims(area1, -1) + tf.expand_dims(area2, -2) - inter_area
    union_area = tf.maximum(union_area, 1e-6)

    iou = inter_area / union_area
    return iou


def from_grid_to_coordinates(p):
    # p = tf.identity(p)
    # shape = tf.shape(p)
    # H, W, Z = shape[0], shape[1], shape[2]
    #
    # x_centers = p[..., 0] * (1.0 / tf.cast(shape[1], p.dtype))
    # y_centers = p[..., 1] * (1.0 / tf.cast(shape[0], p.dtype))
    # width = p[..., 2]
    # height = p[..., 3]
    # confidence = p[..., 4]
    #
    # combined = tf.stack([x_centers, y_centers, width, height, confidence], axis=-1)
    #
    # h_range = tf.range(H, dtype=tf.float32) / tf.cast(H, tf.float32)
    # h_grid = tf.reshape(h_range, [H, 1])
    # h_expanded = tf.expand_dims(h_grid, 1)
    # h_full = tf.tile(h_expanded, [1, W, 1])
    #
    # w_range = tf.range(W, dtype=tf.float32) / tf.cast(W, tf.float32)
    # w_grid = tf.reshape(w_range, [W, 1])
    # w_expanded = tf.expand_dims(w_grid, 0)
    # w_full = tf.tile(w_expanded, [H, 1, 1])
    #
    # merged = tf.concat([w_full, h_full], axis=-1)
    # padding = tf.zeros([H, W, Z - 2])
    # padded = tf.concat([merged, padding], axis=-1)
    #
    # output = combined + padded
    #
    # output = output[output[...,4] >= 0]
    output = cxcywh_to_xyxy(p)
    return output


def cxcywh_to_xyxy(boxes):
    if boxes.shape[0] == 5:
        cx, cy, w, h, confi = tf.split(boxes, 5, axis=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return tf.concat([x1, y1, x2, y2, confi], axis=-1)
    if boxes.shape[0] == 4:
        cx, cy, w, h = tf.split(boxes, 4, axis=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return tf.concat([x1, y1, x2, y2], axis=-1)

