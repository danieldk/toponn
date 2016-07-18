#!/usr/bin/env python

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from enum import Enum


class Phase(Enum):
    train = 1
    validation = 2
    predict = 3


def mask(input, mask_value):
    axis = -1 % len(input.get_shape())
    return tf.reduce_any(
        tf.not_equal(
            input,
            mask_value),
        reduction_indices=axis)

# input: [(batch_size,input_size)], where the list length is the maximum
#        sequence length.
# num_layers: number of RNN layers to stack
# output_size: LSTM output size
# dropout: probability with which to retain outputs, disabled when 1.


def rnn_layers(
        inputs,
        num_layers=1,
        output_size=50,
        dropout=1,
        seq_lens=None,
        bidirectional=False):
    seq_lens = None
    lstm_cell = rnn_cell.BasicLSTMCell(output_size, forget_bias=1.0)
    if dropout < 1:
        lstm_cell = rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=dropout)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
    if bidirectional:
        lstm_bcell = rnn_cell.BasicLSTMCell(output_size, forget_bias=1.0)
        bcell = rnn_cell.MultiRNNCell([lstm_bcell] * num_layers)
        return rnn.bidirectional_rnn(
            cell,
            bcell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_lens)
    else:
        return rnn.rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_lens)


class TopoModel:

    def __init__(
            self,
            config,
            batch_size,
            num_steps,
            input_size,
            phase=Phase.train):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_size = input_size
        size = config.hidden_size
        num_layers = config.num_layers
        num_labels = config.num_labels

        if phase == Phase.train:
            keep_prob = config.keep_prob
            keep_prob_input = config.keep_prob_input
        else:
            keep_prob = 1
            keep_prob_input = 1

        self._input_data = tf.placeholder(
            tf.float32, [batch_size, num_steps, input_size], "inputs")

        if phase in (Phase.train, Phase.validation):
            self._targets = tf.placeholder(
                tf.int32, [batch_size, num_steps], "targets")
            y = tf.transpose(self._targets, [1, 0])
            y = tf.reshape(y, [-1])

        # Shape: (batch_size, num_steps)
        input_mask = mask(self.input_data, 0.0)

        # Get the sequence lengths from the mask, shape: (batch_size)
        seq_lens = tf.reduce_sum(tf.to_int64(input_mask), 1)

        if keep_prob_input < 1:
          inputs = tf.nn.dropout(self._input_data, keep_prob_input)
        else:
          inputs = self._input_data

        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.unpack(inputs)

        # outputs, _, _ = rnn_layers(inputs, num_layers=1, output_size=size,
        #    dropout=keep_prob, seq_lens=seq_lens, bidirectional=True)
        outputs = rnn_layers(
            inputs,
            num_layers=1,
            output_size=size,
            dropout=keep_prob,
            seq_lens=seq_lens,
            bidirectional=True)

        outputs, _ = rnn_layers(outputs, num_layers=1, output_size=size,
                                dropout=keep_prob, seq_lens=seq_lens)

        output = tf.concat(0, outputs)

        softmax_w = tf.get_variable("softmax_w", [size, num_labels])
        softmax_b = tf.get_variable("softmax_b", [num_labels])
        logits = tf.matmul(output, softmax_w) + softmax_b

        if phase in (Phase.train, Phase.validation):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, tf.cast(y, tf.int64))

            # Shape: (num_steps, batch_size)
            loss_mask = tf.transpose(input_mask, [1, 0])
            # Shape: (num_steps * batch_size)
            loss_mask = tf.reshape(loss_mask, [-1])
            loss_mask = tf.cast(loss_mask, tf.float32)

            # Zero out losses for inactive steps.
            losses = tf.mul(loss_mask, losses)

            # Compensate for padded losses.
            losses = tf.truediv(losses, tf.reduce_mean(loss_mask))

            self._loss = loss = tf.reduce_sum(losses) / batch_size

        if phase == Phase.validation:
            _, predicted = tf.nn.top_k(logits)
            predicted = tf.reshape(predicted, [-1])
            correct = tf.equal(predicted, y)
            self._correct = logits
            correct = tf.cast(correct, tf.float32)

            # Zero out correctness for inactive steps.
            correct = tf.mul(loss_mask, correct)

            # Compensate for inactive steps
            correct = tf.truediv(correct, tf.reduce_mean(loss_mask))

            self._accuracy = tf.reduce_mean(correct)
        elif phase == Phase.train:
            self._lr = tf.Variable(0.0, trainable=False)
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
        else:
            _, predicted = tf.nn.top_k(logits)
            predicted = tf.reshape(predicted, [num_steps, batch_size])
            tf.transpose(predicted, name="predictions")

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def correct(self):
        return self._correct

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def targets(self):
        return self._targets
