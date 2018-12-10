#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from enum import Enum

from weight_norm import WeightNorm


class Sharing(Enum):
    none = 1
    initial = 2
    succeeding = 3


def mask_layer(layer, mask):
    return tf.multiply(
        tf.broadcast_to(
            tf.expand_dims(
                mask, -1), tf.shape(layer)), layer)


def dilated_convolution(
        x,
        n_outputs,
        kernel_size,
        n_levels,
        is_training,
        mask,
        glu=False,
        keep_prob=1.0):
    layer = x

    for i in range(n_levels):
        # Only use sharing for layers 1 and up. Layer 0 cannot use sared parameters:
        #
        # - It transforms word embeddings into the hidden representation,
        #   whereas subsequent layers transform hidden representations to
        #   hidden representations.
        # - The input size may differ from the output size.
        if i == 0:
            sharing = Sharing.none
        elif i == 1:
            sharing = Sharing.initial
        else:
            sharing = Sharing.succeeding

        dilation = 2 ** i
        layer = residual_block(
            layer,
            n_outputs,
            kernel_size,
            dilation,
            is_training=is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob,
            sharing=sharing)

    # Mask after last convolution. This is only necessary for models that
    # apply transformations across time steps after the diluted convolutions.
    # But masking is cheap, so better safe than sorry.
    layer = mask_layer(layer, mask)

    return layer


def residual_block(
        x,
        n_outputs,
        kernel_size,
        dilation,
        is_training,
        mask,
        glu=False,
        keep_prob=1.0,
        sharing=Sharing.none):
    if sharing == Sharing.initial or sharing == Sharing.succeeding:
        suffix = "shared"
    else:
        suffix = "unshared"

    with tf.variable_scope("conv1-%s" % suffix, reuse=sharing == Sharing.succeeding):
        conv1 = residual_unit(
            x,
            n_outputs,
            kernel_size,
            dilation,
            is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob)
    with tf.variable_scope("conv2-%s" % suffix, reuse=sharing == Sharing.succeeding):
        conv2 = residual_unit(
            conv1,
            n_outputs,
            kernel_size,
            dilation,
            is_training,
            mask=mask,
            glu=glu,
            keep_prob=keep_prob)

    if x.get_shape()[2] != n_outputs:
        # Note: biases could change padding timesteps, but the next layer will mask
        #       the resulting sequence.
        x = tf.layers.Conv1D(n_outputs, 1)(x)

    return x + conv2


def residual_unit(
        x,
        n_outputs,
        kernel_size,
        dilation,
        is_training,
        mask,
        glu=False,
        keep_prob=1.0):
    if glu:
        # For GLU we need the hidden representation, plus an equal number
        # of parameters for weighting the hidden representation.
        n_outputs *= 2

    # Mask inactive time steps. This is necessary, because convolutions make
    # the padding non-zero (through past timesteps). In later convolutions,
    # these updated paddings would then influence time steps before the
    # padding.
    x = mask_layer(x, mask)
    conv = WeightNorm(
        tf.layers.Conv1D(
            n_outputs,
            kernel_size,
            dilation_rate=dilation,
            padding="same"))(x)

    if glu:
        left, right = tf.split(conv, num_or_size_splits=2, axis=2)
        left = tf.sigmoid(left)
        conv = tf.multiply(left, right)
    else:
        conv = tf.nn.relu(conv)

    # Spatial dropout
    conv = tf.contrib.layers.dropout(
        conv,
        keep_prob=keep_prob,
        noise_shape=[
            tf.shape(conv)[0],
            tf.constant(1),
            tf.shape(conv)[2]],
        is_training=is_training)

    return conv


class ConvModel:
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
            tf.float32, [
                None, None, shapes['token_embed_dims']], name="tokens")
        self._tags = tf.placeholder(
            tf.float32, [
                None, None, shapes['tag_embed_dims']], name="tags")

        # Sequence lenghts
        self._seq_lens = tf.placeholder(
            tf.int32, [None], name="seq_lens")
        input_mask = tf.sequence_mask(self._seq_lens, dtype=tf.float32)

        inputs = tf.concat([self._tokens, self._tags], axis=2)

        inputs = tf.contrib.layers.dropout(
            inputs,
            keep_prob=config.keep_prob_input,
            is_training=self.is_training)

        hidden_states = dilated_convolution(
            inputs,
            config.hidden_size,
            kernel_size=config.kernel_size,
            n_levels=config.n_levels,
            is_training=self.is_training,
            glu=config.glu,
            keep_prob=config.keep_prob,
            mask=input_mask)

        # Normalize hidden layers, seems to speed up convergence.
        hidden_states = tf.contrib.layers.layer_norm(
            hidden_states, begin_norm_axis=-1)

        hidden_states = tf.reshape(hidden_states, [-1, config.hidden_size])

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
            tf.int32, name="predicted")

        correct = tf.equal(predicted, self._labels)

        # Zero out correctness for inactive steps.
        correct = tf.multiply(input_mask, tf.cast(correct, tf.float32))

        # Compensate for inactive steps
        correct = tf.reshape(correct, [-1])
        correct = tf.truediv(correct, tf.reduce_mean(input_mask))

        self._accuracy = tf.reduce_mean(correct, name="accuracy")

        # Optimization with gradient clipping. Consider making the gradient
        # norm a placeholder as well.
        lr = tf.placeholder(tf.float32, [], "lr")
        optimizer = tf.train.AdamOptimizer(lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self._train_op = optimizer.apply_gradients(
            zip(gradients, variables), name="train")

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
