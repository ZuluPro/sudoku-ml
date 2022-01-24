import sys
import numpy as np
from tensorflow.experimental import numpy as tnp
from sklearn.model_selection import train_test_split

from sudoku_game import generators


def norm(a):
    return (a/9)-.5


def denorm(a):
    return (a+.5)*9


class BaseGenerator:
    def generate_dataset(self, count=1000, flatten=False, removed=1):
        raise NotImplementedError()

    def generate_training_dataset(self, count=1000, removed=1):
        raise NotImplementedError()

    def remove(self, y, removed=1):
        x = np.copy(y)
        for i in np.arange(x.shape[0]):
            for _ in np.arange(removed):
                j = np.random.randint(81)
                x[i][j] = 0
        return np.array(x)

    def print_training_dataset(self, dataset, out_fd=None):
        raise NotImplementedError()


class Generator(BaseGenerator):
    np = np

    def __init__(self, processes=4):
        self.processes = processes

    def generate_dataset(self, count=1000, flatten=False, removed=1):
        gen = generators.Generator()
        y = self.np.array(list(gen.generate_solutions(count, processes=self.processes)))
        x = self.remove(y, removed=removed)
        y = y - 1
        if not flatten:
            x = x.reshape((count, 9, 9, 1))
            y = y.reshape((count, 81, 1))
        return x, y


    def generate_training_dataset(self, count=1000, removed=1):
        x, y = self.generate_dataset(
            count=count,
            removed=removed,
        )
        x, y = norm(x), y
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.2,
            random_state=42,
        )
        return x_train, x_test, y_train, y_test

    def print_training_dataset(self, dataset, out_fd=None):
        out_fd = sys.stdout if out_fd is None else out_fd
        size = dataset[0].shape[0]
        x, y = dataset

        x = x.reshape((size, 81)).astype(str)
        y = (y.reshape((size, 81)) + 1).astype(str)
        x = self.np.apply_along_axis(''.join, 1, x)
        y = self.np.apply_along_axis(''.join, 1, y)
        out_fd.write('quizzes,solutions')
        for i in self.np.arange(size):
            out_fd.write('%s,%s' % (x[i], y[i]))


class TfNpGenerator(BaseGenerator):
    np = tnp

    def __init__(self, *args, **kwargs):
        tnp.experimental_enable_numpy_behavior()
        super().__init__(*args, **kwargs)

    def print_training_dataset(self, dataset, out_fd=None):
        out_fd = sys.stdout if out_fd is None else out_fd
        size = dataset[0].shape[0]
        x, y = dataset

        x = x.reshape((size, 81)).numpy().astype(str)
        y = (y.reshape((size, 81)) + 1).numpy().astype(str)
        x = self.np.apply_along_axis(''.join, 1, x)
        y = self.np.apply_along_axis(''.join, 1, y)
        out_fd.write('quizzes,solutions')
        for i in self.np.arange(size):
            out_fd.write('%s,%s' % (x[i], y[i]))
