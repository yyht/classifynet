import tensorflow as tf
import numpy as np

EPSILON = 1e-10

def euclidean(sentence_one, sentence_two):
    sent1 = tf.nn.l2_normalize(sentence_one, axis=-1)
    sent2 = tf.nn.l2_normalize(sentence_two, axis=-1)
    distance = tf.sqrt(tf.reduce_sum((sent1-sent2)**2, axis=-1))
    return distance

def cosine(sentence_one, sentence_two):
    cosine_numerator = tf.reduce_sum(tf.multiply(sentence_one, sentence_two), axis=-1)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(sentence_one), axis=-1),
                             EPSILON))
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(sentence_two), axis=-1),
                             EPSILON))
    
    return cosine_numerator / v1_norm / v2_norm

def arccosine(sentence_one, sentence_two):
    cosine_numerator = tf.reduce_sum(tf.multiply(sentence_one, sentence_two), axis=-1)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(sentence_one), axis=-1),
                             EPSILON))
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(sentence_two), axis=-1),
                             EPSILON))

    distance = cosine_numerator / v1_norm / v2_norm
    distance = tf.clip_by_value(distance, -1, 1)
    distance = tf.acos(distance) / np.pi

    return distance