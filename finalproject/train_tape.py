import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.data import AUTOTUNE

from finalproject.data_loading import process_path
from finalproject.loss import loss_fn
from finalproject.network import Net

grid_h = 1
grid_w = 1
batch_size = 6
epochs = 20
learning_rate = 1e-5

list_ds = tf.data.Dataset.list_files('finalproject/banana-detection/augmented/images/*.png', shuffle=False)
train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

val_list_ds = tf.data.Dataset.list_files('finalproject/banana-detection/bananas_val/images/*.png', shuffle=False)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

network = Net()
network.build(input_shape=(None, 256, 256, 3))

optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,  weight_decay=1e-4)


patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):

    # if epoch == 10 :
    #     learning_rate *= 0.1
    #     optimizer.learning_rate = learning_rate
    #     print(f"Learning rate adjusted to {learning_rate}")

    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss_avg = tf.keras.metrics.Mean()
    pos_losses = tf.keras.metrics.Mean()
    dim_losses = tf.keras.metrics.Mean()

    for step, (images, labels, _) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            predictions = network(images, training=True)
            loss_value, pos_loss, dim_loss = loss_fn(labels[:,0,:], predictions)

        gradients = tape.gradient(loss_value, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))
        max_grad = max([tf.reduce_max(tf.abs(g)) for g in gradients if g is not None])

        # Threshold can be adjusted depending on your model
        if max_grad < 1e-7:
            tf.print("⚠️ Warning: Possible vanishing gradients detected! Max grad =", max_grad)

        epoch_loss_avg.update_state(loss_value)
        pos_losses.update_state(pos_loss)
        dim_losses.update_state(dim_loss)

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss_value.numpy():.4f}")

    val_loss_avg = tf.keras.metrics.Mean()
    for step, (images, labels, _) in enumerate(val_ds):
        predictions = network(images, training=False)
        val_loss, _, _ = loss_fn(labels, predictions)
        val_loss_avg.update_state(val_loss)

    val_loss_value = val_loss_avg.result().numpy()

    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss_avg.result().numpy():.4f}")
    print(f"Epoch {epoch + 1} Position Loss: {pos_losses.result().numpy():.4f}")
    print(f"Epoch {epoch + 1} Dimension Loss: {dim_losses.result().numpy():.4f}")
    print(f"Epoch {epoch + 1} Validation Loss: {val_loss_value:.4f}")

    network.save_weights(f'finalproject/saved_model/epoch_{epoch}.weights.h5')

    if val_loss_value < best_val_loss:
        print("Validation loss improved, saving best model.")
        best_val_loss = val_loss_value
        patience_counter = 0

        network.save_weights('finalproject/saved_model/best_model.weights.h5')

    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
