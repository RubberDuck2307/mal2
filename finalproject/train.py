from data_loading import get_dataset, convert_labels_to_grid_tf
from finalproject.data_loading import load_all_labels

from finalproject.loss import loss_fn
from network import Net
import tensorflow as tf

grid_h = 1
grid_w = 1
batch_size = 8
epochs = 5
learning_rate = 1e-4

labels_dict = load_all_labels("datasets_yolo")
grid = convert_labels_to_grid_tf(labels_dict, grid_h, grid_w)
dataset = get_dataset("datasets/images", grid, batch_size)

network = Net()

network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=loss_fn
)

network.fit(
    dataset,
    epochs=epochs,
)

network.save("saved_model/airplane_detector.keras")
