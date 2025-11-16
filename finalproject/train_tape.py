from data_loading import get_dataset, convert_labels_to_grid_tf
from finalproject.data_loading import load_all_labels
from finalproject.loss import loss_fn
from finalproject.utils import annotate_image
from network import Net
import tensorflow as tf
from PIL import Image

grid_h = 1
grid_w = 1
batch_size = 8
epochs = 10
learning_rate = 1e-5

# Load data
labels_dict = load_all_labels("datasets_yolo")
grid = convert_labels_to_grid_tf(labels_dict, grid_h, grid_w)
dataset = get_dataset("datasets/images", grid, batch_size)

# Initialize model
network = Net()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step, (images, targets) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = network(images, training=True)
            loss_value = loss_fn(targets, predictions)

        gradients = tape.gradient(loss_value, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss_value.numpy():.4f}")




# Save the trained model
network.save("saved_model/airplane_detector.keras")

