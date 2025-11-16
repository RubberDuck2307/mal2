import tensorflow as tf
import math


def get_dataset(image_dir, grid, batch_size):
    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(".jpg")
    ])
    print(image_paths)

    def load_and_preprocess(image_path, label):
        print(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # image = tf.image.resize(image, [640, 360])

        filename = tf.strings.split(image_path, os.sep)[-1]
        name_no_ext = tf.strings.split(filename, ".")[0]

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


def convert_labels_to_grid_tf(labels, grid_h, grid_w):
    batch_size = len(labels)

    Y = tf.zeros((batch_size, grid_h, grid_w, 5), dtype=tf.float32)

    for b in range(batch_size):
        for i in range(len(labels[b])):
            obj = labels[b][i]
            h = math.floor(obj[1] / (1 / grid_h))
            w = math.floor(obj[0] / (1 / grid_w))

            obj_x = (obj[0] / (1 / grid_w)) - w
            obj_y = (obj[1] / (1 / grid_h)) - h
            obj_updated = tf.stack([obj_x, obj_y, obj[2], obj[3], 1])

            indices = [[b, h, w]]
            updates = [obj_updated]
            Y = tf.tensor_scatter_nd_update(Y, indices, updates)
    return Y
