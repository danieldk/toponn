#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from enum import Enum


class Phase(Enum):
    train = 1
    validation = 2
    predict = 3


class Layer(Enum):
    token = 1
    tag = 2


def dropout_wrapper(cell, is_training, keep_prob):
    keep_prob = tf.cond(
        is_training,
        lambda: tf.constant(keep_prob),
        lambda: tf.constant(1.0))
    return tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=keep_prob)


def rnn_layers(
        is_training,
        inputs,
        num_layers=1,
        output_size=50,
        output_dropout=1,
        state_dropout=1,
        seq_lens=None,
        bidirectional=False):
    forward_cell = tf.contrib.rnn.BasicLSTMCell(output_size)
    forward_cell = dropout_wrapper(
        cell=forward_cell,
        is_training=is_training,
        keep_prob=output_dropout)

    if not bidirectional:
        return tf.nn.dynamic_rnn(
            forward_cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_lens)

    backward_cell = tf.contrib.rnn.BasicLSTMCell(output_size)
    backward_cell = dropout_wrapper(
        cell=backward_cell,
        is_training=is_training,
        keep_prob=output_dropout)

    return tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        inputs,
        dtype=tf.float32,
        sequence_length=seq_lens)


class TopoModel:
    def __init__(
            self,
            config,
            shapes):
        # Are we training or not?
        self._is_training = tf.placeholder(tf.bool, [], "is_training")

        self._labels = tf.placeholder(
            tf.int32, name="labels", shape=[
                None, None])

        # Inputs: tags and tokens.
        self._tokens = tf.placeholder(
            tf.int32, [None, None], name="tokens")
        self._tags = tf.placeholder(tf.int32, [None, None], name="tags")
        self._seq_lens = tf.placeholder(
            tf.int32, [None], name="seq_lens")

        # Embeddings
        self._token_embeds = tf.placeholder(
            tf.float32, [None, shapes['token_embed_dims']], "token_embeds")
        self._tag_embeds = tf.placeholder(
            tf.float32, [None, shapes['tag_embed_dims']], "tag_embeds")

        token_embeds = tf.nn.embedding_lookup(self._token_embeds, self._tokens)
        tag_embeds = tf.nn.embedding_lookup(self._tag_embeds, self._tags)

        inputs = tf.concat([token_embeds, tag_embeds], axis=2)

        inputs = tf.contrib.layers.dropout(
            inputs,
            keep_prob=config.keep_prob_input,
            is_training=self.is_training)

        (fstates, bstates), _ = rnn_layers(
            self.is_training,
            inputs,
            num_layers=1,
            output_size=config.hidden_size,
            output_dropout=config.keep_prob,
            state_dropout=config.keep_prob,
            seq_lens=self._seq_lens,
            bidirectional=True)
        hidden_states = tf.concat([fstates, bstates], axis=2)

        hidden_states, _ = rnn_layers(self.is_training, hidden_states, num_layers=1, output_size=config.hidden_size,
                                      output_dropout=config.keep_prob,
                                      state_dropout=config.keep_prob, seq_lens=self._seq_lens)

        hidden_states = tf.reshape(hidden_states, [-1, config.hidden_size])

        hidden_states = batch_norm(
            hidden_states,
            decay=0.98,
            scale=True,
            is_training=self.is_training,
            fused=False,
            updates_collections=None)

        softmax_w = tf.get_variable(
            "softmax_w", [
                config.hidden_size, shapes['n_labels']])
        softmax_b = tf.get_variable("softmax_b", [shapes['n_labels']])
        logits = tf.nn.xw_plus_b(
            hidden_states,
            softmax_w,
            softmax_b,
            name="logits")

        logits = tf.reshape(
            logits, [
                tf.shape(self.tokens)[0], -1, shapes['n_labels']])

        input_mask = tf.sequence_mask(
            self._seq_lens, maxlen=tf.shape(
                self._labels)[1], dtype=tf.float32)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self._labels)

        # Zero out losses for inactive steps.
        losses = tf.multiply(input_mask, losses)

        # Compensate for padded losses.
        losses = tf.reshape(losses, [-1])
        losses = tf.truediv(losses, tf.reduce_mean(input_mask))

        self._loss = loss = tf.reduce_mean(losses, name="loss")

        predicted = tf.cast(
            tf.argmax(
                logits,
                axis=2),
            tf.int32, name = "predicted")

        correct = tf.equal(predicted, self._labels)

        # Zero out correctness for inactive steps.
        correct = tf.multiply(input_mask, tf.cast(correct, tf.float32))

        # Compensate for inactive steps
        correct = tf.reshape(correct, [-1])
        correct = tf.truediv(correct, tf.reduce_mean(input_mask))

        self._accuracy = tf.reduce_mean(correct, name="accuracy")

        lr = tf.placeholder(tf.float32, [], "lr")
        self._train_op = tf.train.AdamOptimizer(lr).minimize(loss, name = "train")

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def correct(self):
        return self._correct

    @property
    def is_training(self):
        return self._is_training

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def labels(self):
        return self._labels

    @property
    def tag_embeds(self):
        return self._tag_embeds

    @property
    def tags(self):
        return self._tags

    @property
    def tokens(self):
        return self._tokens

    @property
    def token_embeds(self):
        return self._token_embeds

    @property
    def seq_lens(self):
        return self._seq_lens
