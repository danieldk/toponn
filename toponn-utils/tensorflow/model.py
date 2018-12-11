import tensorflow as tf


class Model:
    def __init__(self, config, shapes):
        self._config = config
        self._shapes = shapes

    def accuracy(self, prefix, predictions, labels):
        correct = tf.equal(predictions, labels)

        # Mask inactive timesteps
        correct = tf.multiply(self.mask, tf.cast(correct, tf.float32))

        # Compensate for inactive time steps.
        correct = tf.reshape(correct, [-1])
        correct = tf.truediv(correct, tf.reduce_mean(self.mask))

        return tf.reduce_mean(correct, name="%s_accuracy" % prefix)


    def masked_softmax_loss(self, prefix, logits, labels, mask):
        # Compute losses
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        # Mask inactive time steps.
        losses = tf.multiply(mask, losses)

        # Compensate for inactive time steps.
        losses = tf.truediv(losses, tf.reduce_mean(mask))

        # Compute sequence probabilities
        #losses = tf.reduce_sum(losses, axis=1)

        return tf.reduce_mean(losses, name="%s_loss" % prefix)

    def crf_loss(self, prefix, logits, labels):
        with tf.variable_scope("%s_crf" % prefix):
            (loss, transitions) = tf.contrib.crf.crf_log_likelihood(
                logits, labels, self.seq_lens)
        return tf.reduce_mean(-loss, name="%s_loss" % prefix), transitions

    def crf_predictions(self, prefix, logits, transitions):
        predictions, _ = tf.contrib.crf.crf_decode(
            logits, transitions, self.seq_lens)
        return tf.identity(predictions, name="%s_predictions" % prefix)

    def predictions(self, prefix, logits):
        return tf.cast(
            tf.argmax(
                logits,
                axis=2),
            tf.int32, name="%s_predictions" % prefix)

    def setup_placeholders(self):
        self._is_training = tf.placeholder(tf.bool, [], "is_training")

        self._topo_labels = tf.placeholder(
            tf.int32, name="topo_labels", shape=[
                None, None])

        self._tokens = tf.placeholder(
            tf.float32,
            shape=[
                None,
                None,
                self.shapes['token_embed_dims']],
            name="tokens")
        self._tags = tf.placeholder(
            tf.float32, [
                None, None, self.shapes['tag_embed_dims']], name="tags")

        self._seq_lens = tf.placeholder(
            tf.int32, [None], name="seq_lens")

        # Compute mask
        self._mask = tf.sequence_mask(
            self.seq_lens, maxlen=tf.shape(
                self.tags)[1], dtype=tf.float32)

    @property
    def config(self):
        return self._config

    @property
    def is_training(self):
        return self._is_training

    @property
    def mask(self):
        return self._mask

    @property
    def pos_labels(self):
        return self._pos_labels

    @property
    def seq_lens(self):
        return self._seq_lens

    @property
    def tags(self):
        return self._tags

    @property
    def tokens(self):
        return self._tokens

    @property
    def topo_labels(self):
        return self._topo_labels

    @property
    def shapes(self):
        return self._shapes
