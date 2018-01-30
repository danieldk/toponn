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

def rnn_layers(
        inputs,
        num_layers=1,
        output_size=50,
        output_dropout = 1,
        state_dropout = 1,
        seq_lens=None,
        bidirectional=False,
        phase = Phase.train):
    forward_cell = tf.contrib.rnn.BasicLSTMCell(output_size)
    if phase == Phase.train:
        forward_cell = tf.contrib.rnn.DropoutWrapper(
            forward_cell, output_keep_prob=output_dropout)

    if not bidirectional:
        return tf.nn.dynamic_rnn(
            forward_cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_lens)

    backward_cell = tf.contrib.rnn.BasicLSTMCell(output_size)
    if phase == Phase.train:
        backward_cell = tf.contrib.rnn.DropoutWrapper(
            backward_cell, output_keep_prob=output_dropout)

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
            dataset,
            embeddings,
            phase=Phase.train):
        batch_size = config.batch_size
        
        self._labels = tf.placeholder(tf.int32, name="labels", shape=[batch_size, None])

        # Inputs: tags and tokens.
        self._tokens = tf.placeholder(tf.int32, [batch_size, None], name="tokens")
        self._tags = tf.placeholder(tf.int32, [batch_size, None], name="tags")
        self._seq_lens = tf.placeholder(tf.int32, [batch_size], name="seq_lens")

        # Embeddings
        self._token_embeds = tf.placeholder(tf.float32, embeddings[Layer.token].data.shape, "token_embeds")
        self._tag_embeds = tf.placeholder(tf.float32, embeddings[Layer.tag].data.shape, "tag_embeds")

        token_embeds = tf.nn.embedding_lookup(self._token_embeds, self._tokens)
        tag_embeds = tf.nn.embedding_lookup(self._tag_embeds, self._tags)

        inputs = tf.concat([token_embeds, tag_embeds], axis = 2)

        if phase == Phase.train:
          inputs = tf.nn.dropout(inputs, config.input_dropout)

        (fstates, bstates), _ = rnn_layers(
            inputs,
            num_layers=1,
            output_size=config.hidden_size,
            output_dropout = config.hidden_dropout,
            state_dropout = config.hidden_dropout,
            seq_lens=self._seq_lens,
            bidirectional=True,
            phase=phase)
        hidden_states = tf.concat([fstates, bstates], axis=2)


        hidden_states, _ = rnn_layers(hidden_states, num_layers=1, output_size=config.hidden_size,
                                output_dropout = config.hidden_dropout,
                                state_dropout = config.hidden_dropout, seq_lens=self._seq_lens)

        hidden_states = tf.reshape(hidden_states, [-1, config.hidden_size])
        
        hidden_states = batch_norm(hidden_states, scale = True, decay=0.98, fused=False, is_training = phase == Phase.train, reuse = phase != Phase.train, scope = "input_norm", updates_collections=None)

        softmax_w = tf.get_variable("softmax_w", [config.hidden_size, config.num_outputs])
        softmax_b = tf.get_variable("softmax_b", [config.num_outputs])
        logits = tf.nn.xw_plus_b(hidden_states, softmax_w, softmax_b, name = "logits")

        logits = tf.reshape(logits, [config.batch_size, -1, config.num_outputs])

        if phase in (Phase.train, Phase.validation):
            input_mask = tf.sequence_mask(self._seq_lens, maxlen = tf.shape(self._labels)[1], dtype = tf.float32)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits, labels = self._labels)

            # Zero out losses for inactive steps.
            losses = tf.multiply(input_mask, losses)

            # Compensate for padded losses.
            losses = tf.reshape(losses, [-1])
            losses = tf.truediv(losses, tf.reduce_mean(input_mask))

            self._loss = loss = tf.reduce_sum(losses) / config.batch_size

        if phase == Phase.validation:
            predicted = tf.cast(tf.argmax(logits, axis=2, name="predicted"), tf.int32)

            correct = tf.equal(predicted, self._labels)

            # Zero out correctness for inactive steps.
            correct = tf.multiply(input_mask, tf.cast(correct, tf.float32))

            # Compensate for inactive steps
            correct = tf.reshape(correct, [-1])
            correct = tf.truediv(correct, tf.reduce_mean(input_mask))

            self._accuracy = tf.reduce_mean(correct)
        elif phase == Phase.train:
            self._lr = tf.Variable(0.0, trainable=False)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            predicted = tf.cast(tf.argmax(logits, axis=2), tf.int32, name="predicted")
            #predicted = tf.reshape(predicted, [num_steps, batch_size])
            #tf.transpose(predicted, name="predictions")

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def correct(self):
        return self._correct

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

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
