from finalproject.data_loading import load_all_labels, convert_labels_to_grid_tf, get_dataset
from finalproject.loss import from_grid_to_coordinates
from finalproject.train_tape import predictions
from finalproject.utils import annotate_image
from PIL import Image
import tensorflow as tf

grid_h = 1
grid_w = 1
batch_size = 6
epochs = 20
learning_rate = 1e-4

labels_dict = load_all_labels("datasets_yolo")
grid = convert_labels_to_grid_tf(labels_dict, grid_h, grid_w)
dataset = get_dataset("datasets/images", grid, batch_size)

model = tf.keras.models.load_model("saved_model/airplane_detector.keras")


for images, labels in dataset:

    predictions = model(images, training=False)
    label_norm = from_grid_to_coordinates(predictions[0])

    img = Image.fromarray(images[0].numpy())
    annotate_image(img, label_norm, tf.ones([label_norm.shape[0]], dtype=tf.float32), label_norm[..., 4],
                   save_path="output/", file_name="test.jpg", normalized=True)
    break
