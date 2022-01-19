import logging

import numpy as np
import tensorflow as tf

from sudoku_ml import models
from sudoku_ml import datasets


logger = logging.getLogger('sudoku_ml')


class Agent:
    def __init__(
            self,
            batch_size=32,
            epochs=2,
            model_path=None,
            model_save_file=None,
            model_load_file=None,
            log_dir=None,
        ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.model_save_file = model_save_file
        self.model_load_file = model_load_file
        if model_load_file:
            try:
                self.model = self.load_model()
            except OSError:
                self.model = self._compile_model()
        else:
            self.model = self._compile_model()
        self.log_dir = log_dir


    def _compile_model(self):
        model = models.get_model_class(self.model_path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            # metrics=['acc']
        )
        return model

    def train(self, dataset, runs=10, validate_data=True, evaluate=False):
        callbacks = []
        if self.log_dir:
            callbacks + [
                tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
            ]

        dataset_size = dataset[0].shape[0]
        x_train, x_test, y_train, y_test = dataset
        validation_data = (x_test, y_test) if validate_data else None
        for i in range(runs):
            self.model.fit(
                x_train, y_train,
                epochs=self.epochs,
                validation_split=0.2,
                batch_size=self.batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
            )
            if evaluate:
                test_loss = self.model.evaluate(x_test, y_test, verbose=2)

    def infer(self, grid):
        grid = datasets.norm(grid)

        out = self.model.predict(grid.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9))+1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        grid_ = datasets.denorm(grid).reshape((9, 9))
        mask = (grid_==0)

        prob_new = prob*mask
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y] - 1
        return x, y, val

    def save_model(self):
        if self.model_save_file:
            tf.keras.models.save_model(
                model=self.model,
                filepath=self.model_save_file
            )

    def load_model(self):
        logger.debug('Using %s', self.model_load_file)
        return tf.keras.models.load_model(
            filepath=self.model_load_file,
        )
