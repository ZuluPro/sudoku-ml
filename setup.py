from setuptools import setup, find_packages
import sudoku_ml


def read_file(name):
    with open(name) as fd:
        return fd.read()

setup(
    name="sudoku-ml",
    version=sudoku_ml.__version__,
    author=sudoku_ml.__author__,
    author_email=sudoku_ml.__email__,
    description=sudoku_ml.__doc__,
    url=sudoku_ml.__url__,
    license=sudoku_ml.__license__,
    py_modules=['sudoku_ml'],
    packages=find_packages(),
    install_requires=read_file('requirements.txt').splitlines(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        'Operating System :: OS Independent',
        "Programming Language :: Python",
    ],
    long_description=read_file('README.rst'),
    entry_points={'console_scripts': [
        'sudoku-ml = sudoku_ml.console:main',
    ]},
)
