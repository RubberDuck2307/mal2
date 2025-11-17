from finalproject.loss import from_grid_to_coordinates
import tensorflow as tf

prediction = tf.constant([[[0.42109236, 0.45388597, 0.8113333,  0.7703593 , 0.39012972]]])
prediction = from_grid_to_coordinates(prediction)
