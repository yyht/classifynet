import tensorflow as tf
import numpy as np

from metric import pair_wise_metric
import logging

def contrastive_loss(sentence_one, sentence_two, labels, config,
                    *args, **kargs):
    labels = tf.cast(labels, tf.float32)
    if config.metric in ["Euclidean", "Arccosine"]:
        distance = pair_wise_metric.euclidean(sentence_one, sentence_two)
        tmp= labels * tf.square(distance)
        tmp2 = (1-labels) *tf.square(tf.maximum((config.margin - distance), 0))
    elif config.metric in ["Cosine"]:
        distance = pair_wise_metric.cosine(sentence_one, sentence_two)
        tmp= (1-labels) * tf.square(distance)
        tmp2 = (labels) *tf.square(tf.maximum((config.margin - distance), 0))
    return tf.reduce_mean(tmp+tmp2) / 2






