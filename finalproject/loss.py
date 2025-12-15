import tensorflow as tf


def loss_fn(labels, p):
    try:
        x_losses = tf.keras.losses.MeanSquaredError()(labels[..., 0], p[..., 0])
        y_losses = tf.keras.losses.MeanSquaredError()(labels[..., 1], p[..., 1])

        pos_losses = x_losses + y_losses

        h_loss = tf.keras.losses.MeanSquaredError()(labels[..., 2], p[..., 2])
        w_loss = tf.keras.losses.MeanSquaredError()(labels[..., 3], p[..., 3])
        size_losses = h_loss + w_loss

        total_loss = pos_losses + size_losses
        return total_loss, pos_losses, size_losses
    except Exception as e:

        return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)



def from_grid_to_coordinates(p):
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

