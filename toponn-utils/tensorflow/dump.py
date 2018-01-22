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

def usage():
    print("Usage: %s config train_data graph_out" % sys.argv[0])
    sys.exit(1)


def main(unused_args):
    parser = OptionParser()
    (options, args) = parser.parse_args()

    if len(args) != 3:
        usage()

    config = get_config()

    with open(args[0]) as conffile:
        topo_config = toml.loads(conffile.read())

    with DataSet(args[1]) as train_ds:
        config.num_outputs = train_ds.max_label() + 1

        embeds_conf = topo_config["embeddings"]
        token_embeds = Embeddings(path_relative_to_conf(args[0], embeds_conf["word"]["filename"]))
        tag_embed = Embeddings(path_relative_to_conf(args[0], embeds_conf["tag"]["filename"]))

        embeddings = {
                Layer.token: token_embeds,
                Layer.tag: tag_embed,
        }

    tfconfig = tf.ConfigProto()
    with tf.Graph().as_default(), tf.Session(config=tfconfig) as session:
        with tf.name_scope("training"):
            with tf.variable_scope("model", reuse=None):
                m = TopoModel(
                    config=config,
                    phase=Phase.train,
                    dataset=train_ds,
                    embeddings=embeddings)

        with tf.name_scope("prediction"):
            with tf.variable_scope("model", reuse=True):
                mvalid = TopoModel(
                    config=config,
                    phase=Phase.predict,
                    dataset=train_ds,
                    embeddings=embeddings)

        tf.train.write_graph(
            session.graph.as_graph_def(),
            './',
            args[2])

if __name__ == "__main__":
    tf.app.run()
