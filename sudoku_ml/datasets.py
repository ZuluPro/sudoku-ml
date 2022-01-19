import numpy as np
from sklearn.model_selection import train_test_split
from sudoku_game import generators


def norm(a):
    return (a/9)-.5


def denorm(a):
    return (a+.5)*9


def generate_dataset(count=1000, flatten=False, removed=1):
    gen = generators.Generator()
    ys = np.array(list(gen.generate_solutions(count)))
    xs = ys.copy()
    for x in xs:
        for i in range(removed):
            j = np.random.choice(np.arange(81))
            x[j] = 0
    ys = ys - 1
    if not flatten:
        xs = xs.reshape((count, 9, 9, 1))
        ys = ys.reshape((count, 81, 1))
    return xs, ys


def generate_training_dataset(count=1000, removed=1):
    x, y = generate_dataset(count=count, removed=removed)
    x, y = norm(x), y
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
    )
    return x_train, x_test, y_train, y_test
