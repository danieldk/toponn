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


def usage():
    print("Usage: %s config train_data weights graph_out" % sys.argv[0])
    sys.exit(1)


def main(unused_args):
    if len(unused_args) != 5:
        usage()

    config = get_config()

    predict_config = get_config()
    predict_config.batch_size = 1

    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session, \
            h5py.File(unused_args[2], 'r') as train_data:
        data_shape = train_data['batch0']['inputs'].shape
        batch_size = data_shape[0]
        num_steps = data_shape[1]
        input_size = data_shape[2]

        with tf.variable_scope("model", reuse=None):
            m = TopoModel(
                phase=Phase.train,
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                input_size=input_size)

        with tf.variable_scope("model", reuse=True):
            mvalid = TopoModel(
                phase=Phase.validation,
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                input_size=input_size)

        with tf.variable_scope("model", reuse=True):
            mpredict = TopoModel(
                phase=Phase.predict,
                config=config,
                batch_size=batch_size,
                num_steps=num_steps,
                input_size=input_size)

        saver = tf.train.Saver()

        tf.initialize_all_variables().run()

        saver.restore(session, unused_args[3])

        tf.train.write_graph(
            session.graph.as_graph_def(),
            '/tmp/my-model',
            unused_args[4])

if __name__ == "__main__":
    tf.app.run()
