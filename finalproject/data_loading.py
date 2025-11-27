import tensorflow as tf
import math
from finalproject.loss import cxcywh_to_xyxy

from tensorflow.python.data import AUTOTUNE


def get_dataset(image_dir, grid, batch_size):
    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(".jpg")
    ])

    def load_and_preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.image.resize(image, [640, 360])

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, grid))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


import os


def load_all_labels(label_folder):
    labels = []

    for fname in sorted(os.listdir(label_folder)):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(label_folder, fname)
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    boxes.append(parts[1:5])

        labels.append(boxes)

    return labels


def look_up_labels(file_names, labels):
    return [labels[k] for k in file_names]


def convert_labels_to_grid_tf(labels, grid_h=1, grid_w=1):
    Y = tf.zeros((grid_h, grid_w, 5), dtype=tf.float32)

    size = tf.shape(labels)[0]
    if size == 0:
        return Y
    for i in range(size):
        try:
            obj = labels[i]
        except:
            break
        h = math.floor(obj[1] / (1 / grid_h))
        w = math.floor(obj[0] / (1 / grid_w))

        obj_x = (obj[0] / (1 / grid_w)) - w
        obj_y = (obj[1] / (1 / grid_h)) - h
        obj_updated = tf.stack([obj_x, obj_y, obj[2], obj[3], 1])

        indices = [[h, w]]
        updates = [obj_updated]
        Y = tf.tensor_scatter_nd_update(Y, indices, updates)
    return Y


def get_label(file_path):
    file_path = file_path.numpy().decode('utf-8')
    file_path = os.path.splitext(file_path)[0]
    file_name = os.path.basename(file_path)
    folder = os.path.dirname(file_path)
    label_path = f"{folder}/labels/{file_name}.txt"
    boxes = []
    if os.path.isdir(label_path):
        return tf.convert_to_tensor(boxes, dtype=tf.float32)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    boxes.append(parts[1:5])
    return tf.convert_to_tensor(boxes, dtype=tf.float32)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    boxes = tf.py_function(func=get_label, inp=[file_path], Tout=tf.float32)
    # boxes = tf.py_function(func=convert_labels_to_grid_tf, inp=[boxes], Tout=tf.float32)
    return img, boxes, file_path


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.batch(4)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
