import tensorflow as tf
from utils import ctc_decode, tokens2sparse, sparse2dense


class SequenceAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False, 
        name = 'seq_acc', 
        **kwargs
    ):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        sparse_true = tokens2sparse(y_true)
        sparse_pred = tokens2sparse(y_pred)

        y_true = sparse2dense(sparse_true, [batch_size, max_length])
        y_pred = sparse2dense(sparse_pred, [batch_size, max_length])

        num_errors = tf.reduce_any(y_true != y_pred, axis=1)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.cast(batch_size, tf.float32)

        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False,
        name = 'char_acc', 
        **kwargs
    ):
        super(CharacterAccuracy, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        num_errors = tf.logical_and(y_true != y_pred, y_true != 0)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))

        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


class LevenshteinDistance(tf.keras.metrics.Metric):
    def __init__(
        self, 
        use_ctc_decode = False,
        normalize = False,
        name = 'levenshtein_distance', 
        **kwargs
    ):
        super(LevenshteinDistance, self).__init__(name=name, **kwargs)
        self.use_ctc_decode = use_ctc_decode
        self.normalize = normalize
        self.sum_distance = self.add_weight(name='sum_distance', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        if self.use_ctc_decode: y_pred = ctc_decode(y_pred, max_length)

        sparse_true = tokens2sparse(y_true)
        sparse_pred = tokens2sparse(y_pred)

        edit_distances = tf.edit_distance(sparse_pred, sparse_true, normalize=self.normalize)
        self.sum_distance.assign_add(tf.reduce_sum(edit_distances))
        self.total.assign_add(tf.cast(batch_size, tf.float32))
    
    def result(self):
        return tf.math.divide_no_nan(self.sum_distance, self.total)

    def reset_state(self):
        self.sum_distance.assign(0)
        self.total.assign(0)
