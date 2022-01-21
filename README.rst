Tensorflow Sudoku Solver
========================

Tensorflow 2 utilities to generate datasets, train a model and solve Sudokus.

This repository is the core of `Sudoku Machine Learning Benchmark <https://github.com/cloudmercato/sudoku-ml-benchmark>`_.

Features
========

- Datasets

  - Generate datasets for training, validation and inference
  - Set the number of hidden numbers in grid
  
- Model

  - Save and load trained model
  - Define you own neural network

- Monitoring

  - Logging management
  - Tensorboard and debuggers management

Install
-------

::

  pip install https://github.com/cloudmercato/sudoku-ml/archive/refs/heads/master.zip
  
Usage
-----

:: 

  $ sudoku-ml --help
  
  usage: sudoku-ml [-h] [--runs RUNS] [--batch-size BATCH_SIZE]
                   [--epochs EPOCHS] [--dataset-size DATASET_SIZE]
                   [--generator-processes GENERATOR_PROCESSES]
                   [--model-path MODEL_PATH] [--model-load-file MODEL_LOAD_FILE]
                   [--model-save-file MODEL_SAVE_FILE] [--log-dir LOG_DIR]
                   [--tf-log-device] [--tf-dump-debug-info]
                   [--tf-profiler-port TF_PROFILER_PORT] [--verbose VERBOSE]
                   [--tf-verbose TF_VERBOSE]
                   [{train,infer,generate}]

  positional arguments:
    {train,infer,generate}

  optional arguments:
    -h, --help            show this help message and exit
    --runs RUNS           Number of runs
    --batch-size BATCH_SIZE
    --epochs EPOCHS
    --dataset-size DATASET_SIZE
    --generator-processes GENERATOR_PROCESSES
    --model-path MODEL_PATH
                          Python path to the model to compile
    --model-load-file MODEL_LOAD_FILE
                          Model load file path (h5)
    --model-save-file MODEL_SAVE_FILE
                          Model save file path (h5)
    --log-dir LOG_DIR     Tensorboard log directory
    --tf-log-device       Determines whether TF compute device info is
                          displayed.
    --tf-dump-debug-info
    --tf-profiler-port TF_PROFILER_PORT
    --verbose VERBOSE, -v VERBOSE
    --tf-verbose TF_VERBOSE, -tfv TF_VERBOSE
  
  
Docker support
--------------

Dockerfile for classic Tensorflow and the GPU version are available: ::

  # For CPU
  docker build -f Dockerfile -t sudoku-ml .
  docker run -it sudoku-ml
  
  # For GPU
  docker build -f Dockerfile-gpu -t sudoku-ml-gpu .
  docker run --gpus all -it sudoku-ml-gpu
  
The commands above will run a training, then save the model in `/models/current.h5`. You can mount a volume on `/models/` to keep it. In the same idea you can mount a volume on `/log_dir/`, to retrive the Tensorboard data.

    
    

