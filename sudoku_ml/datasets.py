from tensorflow.experimental import numpy as np
from sklearn.model_selection import train_test_split
from sudoku_game import generators

np.experimental_enable_numpy_behavior()



def norm(a):
    return (a/9)-.5


def denorm(a):
    return (a+.5)*9


def remove(y, removed=1):
    x = np.copy(y).numpy()
    for i in np.arange(x.shape[0]):
        for _ in np.arange(removed):
            j = np.random.randint(81)
            x[i][j] = 0
    return np.array(x)


def generate_dataset(count=1000, flatten=False, removed=1, processes=4):
    gen = generators.Generator()
    y = np.array(list(gen.generate_solutions(count, processes=processes)))
    x = remove(y, removed=removed)
    y = y - 1
    if not flatten:
        x = x.reshape((count, 9, 9, 1))
        y = y.reshape((count, 81, 1))
    return x, y


def generate_training_dataset(count=1000, removed=1, processes=4):
    x, y = generate_dataset(
        count=count,
        removed=removed,
        processes=processes,
    )
    x, y = norm(x), y
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
    )
    return x_train, x_test, y_train, y_test
