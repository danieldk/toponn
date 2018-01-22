#!/usr/bin/env python

import itertools
from optparse import OptionParser
import sys

import h5py
import tensorflow as tf
import toml

from config import DefaultConfig, path_relative_to_conf
from data import DataSet, Embeddings
from model import Layer, Phase, TopoModel

flags = tf.flags
logging = tf.logging
flags.DEFINE_string(
    "model", "default",
    "The type of model, currently only 'default' is available.")
FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == "default":
        return DefaultConfig()

def run_epoch(session, m, config, dataset, eval_op, embeddings, verbose=False, accuracy=False):
    """Runs the model on the given data."""
    losses = 0.0
    accs = 0.0
    batch = 0

    n_batches = dataset.n_batches

    for step in range(n_batches):
        tokens = dataset.batch_layer(step, 'tokens')
        tags = dataset.batch_layer(step, 'tags')
        labels = dataset.batch_layer(step, 'labels')
        seq_lens = dataset.batch_layer(step, 'lens')

        if accuracy:
            cost, acc, _ = session.run([
                m.loss,
                m.accuracy,
                eval_op],
            {
                m.tokens: tokens,
                m.tags: tags,
                m.labels: labels,
                m.seq_lens: seq_lens,
                m.token_embeds: embeddings[Layer.token].data,
                m.tag_embeds: embeddings[Layer.tag].data
            })
            accs += acc
        else:
            cost, _ = session.run([
                m.loss,
                eval_op],
            {
                m.tokens: tokens,
                m.tags: tags,
                m.labels: labels,
                m.seq_lens: seq_lens,
                m.token_embeds: embeddings[Layer.token].data,
                m.tag_embeds: embeddings[Layer.tag].data
            })

        losses += cost
        batch += 1

    accs /= n_batches
    losses /= n_batches
    return losses, accs


def usage():
    print("Usage: %s train validation" % sys.argv[0])
    sys.exit(1)

def main(unused_args):
    parser = OptionParser()
    (options, args) = parser.parse_args()

    if len(args) != 3:
        usage()

    config = get_config()

    with open(args[0]) as conffile:
        topo_config = toml.loads(conffile.read())

    with DataSet(args[1]) as train_ds,\
            DataSet(args[2]) as validation_ds:
        config.num_outputs = train_ds.max_label() + 1

        embeds_conf = topo_config["embeddings"]
        token_embeds = Embeddings(path_relative_to_conf(args[0], embeds_conf["word"]["filename"]))
        tag_embed = Embeddings(path_relative_to_conf(args[0], embeds_conf["tag"]["filename"]))

        embeddings = {
                Layer.token: token_embeds,
                Layer.tag: tag_embed,
        }

        train_model(
            config=config,
            train_ds=train_ds,
            validation_ds=validation_ds,
            embeddings=embeddings)


def train_model(
        config,
        train_ds,
        validation_ds,
        embeddings):
    gpuopts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tfconfig = tf.ConfigProto(gpu_options=gpuopts)

    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("training"):
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = TopoModel(
                    config=config,
                    phase=Phase.train,
                    dataset=train_ds,
                    embeddings=embeddings)

        with tf.name_scope("validation"):
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = TopoModel(
                    config=config,
                    phase=Phase.validation,
                    dataset=train_ds,
                    embeddings=embeddings)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=0)

        for i in range(config.max_epoch):
            lr = 0.05 * (1 + 0.02 * (i + 1)) ** -2
            #lr = 0.05
            m.assign_lr(session, lr)

            train_loss, _ = run_epoch(session, m, config, train_ds, m.train_op,
                                      embeddings, verbose=True, accuracy=False)

            valid_loss, accs = run_epoch(
                session, mvalid, config, validation_ds, tf.no_op(), embeddings, accuracy=True)
            print("Epoch: %d, lr: %.3f, train loss: %.3f, valid loss: %.3f, acc: %.3f" % (i + 1, session.run(m.lr), train_loss, valid_loss, accs))

            save_path = saver.save(session, "./model", global_step=i + 1)

if __name__ == "__main__":
    tf.app.run()
