import tensorflow as tf


def loss_fn(labels, p):
    object_mask = labels[..., 4] == 1

    p_filtered = p[object_mask]
    labels_filtered = labels[object_mask]

    x_losses = tf.keras.losses.MeanSquaredError()(labels_filtered[..., 0], p_filtered[..., 0])
    y_losses = tf.keras.losses.MeanSquaredError()(labels_filtered[..., 1], p_filtered[..., 1])

    pos_losses = x_losses + y_losses

    h_loss = tf.keras.losses.MeanSquaredError()(labels_filtered[..., 2], p_filtered[..., 2])
    w_loss = tf.keras.losses.MeanSquaredError()(labels_filtered[..., 3], p_filtered[..., 3])
    size_losses = h_loss + w_loss

    confidence_object_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(labels_filtered[..., 4]),
                                                                  p_filtered[..., 4])

    no_object_mask = labels[..., 4] == 0
    p_noobj = p[no_object_mask]
    labels_noobj = labels[no_object_mask]
    confidence_no_object_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(labels_noobj[..., 4]),
                                                                     p_noobj[..., 4])

    total_loss = pos_losses + size_losses
    return total_loss, pos_losses, size_losses


def from_grid_to_coordinates(p):
    p = tf.identity(p)
    shape = tf.shape(p)
    H, W, Z = shape[0], shape[1], shape[2]

    x_centers = p[..., 0] * (1.0 / tf.cast(shape[1], p.dtype))
    y_centers = p[..., 1] * (1.0 / tf.cast(shape[0], p.dtype))
    width = p[..., 2]
    height = p[..., 3]
    confidence = p[..., 4]

    combined = tf.stack([x_centers, y_centers, width, height, confidence], axis=-1)

    h_range = tf.range(H, dtype=tf.float32) / tf.cast(H, tf.float32)
    h_grid = tf.reshape(h_range, [H, 1])
    h_expanded = tf.expand_dims(h_grid, 1)
    h_full = tf.tile(h_expanded, [1, W, 1])

    w_range = tf.range(W, dtype=tf.float32) / tf.cast(W, tf.float32)
    w_grid = tf.reshape(w_range, [W, 1])
    w_expanded = tf.expand_dims(w_grid, 0)
    w_full = tf.tile(w_expanded, [H, 1, 1])

    merged = tf.concat([w_full, h_full], axis=-1)
    padding = tf.zeros([H, W, Z - 2])
    padded = tf.concat([merged, padding], axis=-1)

    output = combined + padded

    output = tf.reshape(output, [-1, 5])
    output = cxcywh_to_xyxy(output)
    return output

def cxcywh_to_xyxy(boxes):
    if boxes.shape[-1] == 5:
        cx, cy, w, h, confi = tf.split(boxes, 5, axis=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return tf.concat([x1, y1, x2, y2, confi], axis=-1)
    elif boxes.shape[-1] == 4:
        cx, cy, w, h = tf.split(boxes, 4, axis=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return tf.concat([x1, y1, x2, y2], axis=-1)
