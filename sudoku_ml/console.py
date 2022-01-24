import os
import argparse
import logging

import numpy as bnp
import tensorflow as tf
from tensorflow.experimental import numpy as np

from sudoku_ml.agent import Agent
from sudoku_ml import datasets
from sudoku_ml import utils


logger = logging.getLogger('sudoku_ml')
tf_logger = logging.getLogger('tensorflow')
    

parser = argparse.ArgumentParser()
# Common
parser.add_argument('action', default='train', choices=('train', 'infer', 'generate'), nargs='?')
parser.add_argument('--runs', type=int, default=100, help='Number of runs')
# Training
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=32)
# Inference
# Dataset
parser.add_argument('--generator-processes', type=int, default=4)
parser.add_argument('--dataset-path', type=str, default=None)
parser.add_argument('--dataset-size', type=int, default=100000)
parser.add_argument('--dataset-removed', type=str, default='10,50')
# Model
parser.add_argument('--model-path', default='sudoku_ml.models.DEFAULT_MODEL',
                    help='Python path to the model to compile')
parser.add_argument('--model-load-file', default=None,
                    help='Model load file path (h5)')
parser.add_argument('--model-save-file', default='model.h5',
                    help='Model save file path (h5)')
parser.add_argument('--log-dir', default=None,
                    help='Tensorboard log directory')
parser.add_argument('--tf-log-device', default=False, action="store_true",
                    help='Determines whether TF compute device info is displayed.')
parser.add_argument('--tf-dump-debug-info', default=False, action="store_true")
parser.add_argument('--tf-profiler-port', default=0, type=int)
parser.add_argument('--verbose', '-v', default=3, type=int)
parser.add_argument('--tf-verbose', '-tfv', default=2, type=int)


def main():
    args = parser.parse_args()

    log_verbose = 60 - (args.verbose*10)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_verbose)
    logger.addHandler(log_handler)
    logger.setLevel(log_verbose)

    tf_log_verbose = 60 - (args.tf_verbose*10)
    tf_logger.setLevel(tf_log_verbose)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', str(args.tf_verbose))

    logger.debug('Config: %s', vars(args))

    tf.debugging.set_log_device_placement(args.tf_log_device)
    if args.log_dir and args.tf_dump_debug_info:
        tf.debugging.experimental.enable_dump_debug_info(
            args.log_dir,
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1
        )
    if args.tf_profiler_port:
        tf.profiler.experimental.server.start(args.tf_profiler_port)

    removed = utils.parse_remove(args.dataset_removed)
    agent = Agent(
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_path=args.model_path,
        model_load_file=args.model_load_file,
        model_save_file=args.model_save_file,
        log_dir=args.log_dir,
        verbose=args.tf_verbose,
    )
    if args.dataset_path:
        generator = datasets.FromFileGenerator(
            fd=args.dataset_path,
        )
    else:
        generator = datasets.Generator(
            processes=args.generator_processes,
        )

    if args.action == 'train':
        dataset = generator.generate_training_dataset(
            count=args.dataset_size,
            removed=removed,
        )
        agent.train(
            runs=args.runs,
            dataset=dataset,
        )
        agent.save_model()

    elif args.action == 'infer':
        valid_count = 0
        x, y = generator.generate_dataset(
            count=args.runs,
            removed=removed,
        )
        for i in range(args.runs):
            X, Y, value = agent.infer(x[i])
            is_valid = y[i].reshape((9, 9))[X, Y] == value
            valid_count += bool(is_valid)
            logger.info('%s\t: %s - %s', i, (X, Y, value+1), is_valid)
            logger.debug('%s', x[i].reshape((9, 9)))

        print('Success: %s/%s' % (valid_count, args.runs))

    elif args.action == 'generate':
        x, y = generator.generate_dataset(
            count=args.dataset_size,
            removed=removed,
        )
        generator.print_training_dataset((x, y))


if __name__ == "__main__":
    main()
