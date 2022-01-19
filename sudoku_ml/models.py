import tensorflow as tf
from sudoku_ml import utils


model_conv2d_relu_softmax = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9*81),
    tf.keras.layers.Reshape((-1, 9)),
    tf.keras.layers.Activation('softmax'),
])

DEFAULT_MODEL = model_conv2d_relu_softmax


def get_model_class(path=None):
    if path is None:
        return DEFAULT_MODEL
    return utils.import_obj(path)
