#!/usr/bin/env python

import itertools
import sys

import h5py
import tensorflow as tf
from keras.utils import generic_utils

from model import Phase, TopoModel

flags = tf.flags
logging = tf.logging
flags.DEFINE_string(
    "model", "default",
    "The type of model, currently only 'default' is available.")
FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == "default":
        return DefaultConfig()


class DefaultConfig:
    init_scale = 0.05
    hidden_size = 50
    keep_prob = 0.90
    keep_prob_input = 0.90
    num_layers = 2
    num_labels = 10
    max_epoch = 30
    lr_decay = 0.8
    learning_rate = 1.0
    max_grad_norm = 5


def run_epoch(session, m, data, eval_op, verbose=False, accuracy=False):
    """Runs the model on the given data."""
    losses = 0.0
    accs = 0.0
    batch = 0

    batches = list(data)

    pb = generic_utils.Progbar(len(batches))

    for step, batch_name in enumerate(batches):
        x = data[batch_name]['inputs']
        y = data[batch_name]['labels']

        if accuracy:
            cost, acc, correct, _ = session.run([m.loss, m.accuracy, m.correct, eval_op],
                                                {m.input_data: x, m.targets: y})
            accs += acc
        else:
            cost, _ = session.run([m.loss, eval_op],
                                  {m.input_data: x, m.targets: y})

        losses += cost
        batch += 1

        pb.update(batch)

    accs /= len(batches)
    losses /= len(batches)
    return losses, accs


def usage():
    print("Usage: %s train validation" % sys.argv[0])
    sys.exit(1)


def main(unused_args):
    if len(unused_args) != 3:
        usage()

    config = get_config()

    with h5py.File(unused_args[1], 'r') as train_data,\
            h5py.File(unused_args[2], 'r') as validation_data:

        data_shape = train_data['batch0']['inputs'].shape
        batch_size = data_shape[0]
        num_steps = data_shape[1]
        input_size = data_shape[2]

        print("Batch size: %d" % batch_size)
        print("Timesteps: %d" % num_steps)
        print("Input size: %d" % input_size)

        train_model(
            config,
            batch_size,
            num_steps,
            input_size,
            train_data,
            validation_data)


def train_model(
        config,
        batch_size,
        num_steps,
        input_size,
        train_data,
        validation_data):
    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = TopoModel(
                phase=Phase.train,
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                input_size=input_size)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = TopoModel(
                phase=Phase.validation,
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                input_size=input_size)

        tf.initialize_all_variables().run()

        saver = tf.train.Saver(max_to_keep=0)

        for i in range(config.max_epoch):
            lr = 0.01 * (1 + 0.02 * (i + 1)) ** -2
            m.assign_lr(session, lr)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_loss, _ = run_epoch(session, m, train_data, m.train_op,
                                      verbose=True, accuracy=False)
            print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))

            valid_loss, accs = run_epoch(
                session, mvalid, validation_data, tf.no_op(), accuracy=True)
            print("Epoch: %d Validation loss: %.3f" % (i + 1, valid_loss))
            print("Epoch: %d Validation accuracy: %.3f" % (i + 1, accs))

            save_path = saver.save(session, "./model", global_step=i + 1)

if __name__ == "__main__":
    tf.app.run()
