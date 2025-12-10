import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from finalproject.data_loading import process_path
from finalproject.loss import loss_fn
from finalproject.network import Net

batch_size = 1
epochs = 100
learning_rate = 1e-2

list_ds = tf.data.Dataset.list_files('banana-detection/single/*.png', shuffle=False)
train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

val_list_ds = tf.data.Dataset.list_files('banana-detection/bananas_val/images/*.png', shuffle=False)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

network = Net()
network.build(input_shape=(None, 256, 256, 3))
# network.load_weights('saved_model/my_checkpoint.weights.h5')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


patience = 2
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    if epoch in [20,50] and epoch != 0:
        learning_rate *= 0.5
        optimizer.learning_rate = learning_rate
        print(f"Learning rate adjusted to {learning_rate}")

    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss_avg = tf.keras.metrics.Mean()
    pos_losses = tf.keras.metrics.Mean()
    dim_losses = tf.keras.metrics.Mean()

    for step, (images, labels, _) in tqdm(enumerate(train_ds)):
        with tf.GradientTape() as tape:
            predictions = network(images, training=True)
            loss_value, pos_loss, dim_loss = loss_fn(labels, predictions)

            gradients = tape.gradient(loss_value, network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        pos_losses.update_state(pos_loss)
        dim_losses.update_state(dim_loss)

    val_loss_avg = tf.keras.metrics.Mean()
    # for step, (images, labels, _) in enumerate(val_ds):
    #     predictions = network(images, training=False)
    #     val_loss, _, _ = loss_fn(labels, predictions)
    #     val_loss_avg.update_state(val_loss)
    #
    # val_loss_value = val_loss_avg.result().numpy()

    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss_avg.result().numpy():.4f}")
    print(f"Epoch {epoch + 1} Position Loss: {pos_losses.result().numpy():.4f}")
    print(f"Epoch {epoch + 1} Dimension Loss: {dim_losses.result().numpy():.4f}")
    # print(f"Epoch {epoch + 1} Validation Loss: {val_loss_value:.4f}")
    #
    #
    # if val_loss_value < best_val_loss:
    #     print("Validation loss improved, saving best model.")
    #     best_val_loss = val_loss_value
    #     patience_counter = 0
    #
    network.save_weights('saved_model/best_model.weights.h5')
    #
    # else:
    #     patience_counter += 1
    #     print(f"No improvement. Patience counter: {patience_counter}/{patience}")
    #
    #     if patience_counter >= patience:
    #         print("Early stopping triggered!")
    #         break
