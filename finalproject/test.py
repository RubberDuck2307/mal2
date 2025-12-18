import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from torchmetrics.detection import mean_ap
from tqdm import tqdm

from finalproject.data_loading import process_path
from finalproject.network import Net
from finalproject.loss import from_grid_to_coordinates, cxcywh_to_xyxy
import PIL.Image
from finalproject.utils import annotate_image
import torch

os.chdir("/home/user/src_wsl/mal2")


def eval(mean_ap, detection, label):
    det_boxes = torch.tensor(detection.numpy(), dtype=torch.float32)
    det_scores = torch.tensor([1.0], dtype=torch.float32)
    det_labels = torch.tensor([1], dtype=torch.int64)

    label_boxes = torch.tensor(label.numpy(), dtype=torch.float32)
    label_labels = torch.tensor([1], dtype=torch.int64)

    eval_detection = {
        "boxes": det_boxes.unsqueeze(0),
        "scores": det_scores,
        "labels": det_labels
    }

    eval_label = {
        "boxes": label_boxes.unsqueeze(0),
        "labels": label_labels
    }

    mean_ap.update([eval_detection], [eval_label])


with tf.device('/CPU:0'):
    batch_size = 1
    epochs = 20

    list_ds = tf.data.Dataset.list_files('finalproject/banana-detection/bananas_val/images/*.png', shuffle=False)
    val_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    ap = mean_ap.MeanAveragePrecision()
    model = Net()
    model.build((None, 256, 256, 3))
    model.load_weights('finalproject/saved_model/best_model.weights.h5')

    for i, (images, labels, file_path) in tqdm(enumerate(val_ds)):
        prediction = model.predict(images)
        prediction = from_grid_to_coordinates(prediction)
        if i % 5 == 0:
            img = PIL.Image.open(file_path[0].numpy().decode('utf-8'))
            img = annotate_image(img, targets=tf.expand_dims(prediction, 0), scores=tf.constant([0]), labels=tf.constant([0]),
                                 normalized=True)
            plt.figure(figsize=(4, 4), facecolor='black')
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        eval(ap, prediction[0:4], cxcywh_to_xyxy(labels[0][0]))

    print(ap.compute())
