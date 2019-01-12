import tensorflow as tf
import numpy as np

import logging

EPSILON = 1e-8

def focal_loss_binary_v1(logits, labels, *args, **kargs):
    """Compute sigmoid focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
    onehot_labels: Each row labels[i] must be a valid probability distribution
    cls_preds: Unscaled log probabilities
    alpha: The hyperparameter for adjusting biased samples, default is 0.25
    gamma: The hyperparameter for penalizing the easy labeled samples
    name: A name for the operation (optional)
    Returns:
    A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    config = args[0]
    alpha = config.alpha
    gamma = config.gamma
    name = config.name
    scope = config.scope
    with tf.name_scope(scope, 'focal_loss') as sc:

        precise_logits = tf.cast(logits, tf.float32)
        onehot_labels = tf.cast(labels, precise_logits.dtype)

        onehot_labels = tf.cast(tf.expand_dims(onehot_labels, -1), tf.int32)

        preds = tf.nn.softmax(precise_logits)
        batch_idxs = tf.range(0, tf.shape(onehot_labels)[0])
        batch_idxs = tf.expand_dims(batch_idxs, 1)

        idxs = tf.concat([batch_idxs, onehot_labels], 1)
        predictions = tf.gather_nd(preds, idxs)
        
        onehot_labels = tf.squeeze(onehot_labels, axis=-1)
        # add small value to avoid 0

        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1), alpha_t, 1-alpha_t)
        losses = -alpha_t * tf.pow(1. - predictions, gamma) * tf.log(predictions+EPSILON)
        return tf.reduce_mean(losses), preds

def focal_loss_binary_v2(logits, labels, *args, **kargs):
    """
    alpha = 0.5
    gamma = 2.0
    """
    config = args[0]
    alpha = config.alpha
    gamma = config.gamma
    name = config.name
    scope = config.scope

    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

    predictions = tf.nn.softmax(logits)
    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)

    labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

    postive_loss = labels * tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)* alpha
    negative_loss = (1-labels)*tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma) * (1 - alpha)

    losses = -postive_loss - negative_loss
    return tf.reduce_mean(losses), predictions

def sparse_amsoftmax_loss(logits, labels, *args, **kargs):
    """
    scale = 30,
    margin = 0.35
    """
    config = args[0]
    scale = config.scale
    margin =config.margin

    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

    y_pred = tf.nn.l2_normalize(logits, axis=-1)
    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)

    y_true_pred = tf.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin
    _Z = tf.concat([y_pred, y_true_pred_margin], 1) 
    _Z = _Z * scale 
    logZ = tf.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + tf.log(1 - tf.exp(scale * y_true_pred - logZ) + EPSILON)
    losses = y_true_pred_margin * scale - logZ
    return -tf.reduce_mean(losses), y_pred

def focal_loss_multi_v1(logits, labels, *args, **kargs):
    config = args[0]
    gamma = config.gamma

    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)

    predictions = tf.nn.softmax(logits)

    # predictions = tf.exp(tf.nn.log_softmax(logits))

    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)

    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)

    labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)

    losses =  tf.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)

    return -tf.reduce_mean(losses), predictions

def softmax_loss(logits, labels, *args, **kargs):
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.int32)
    losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                        labels=labels))
    return losses, tf.nn.softmax(logits)

# def softmax_loss_v1(logits, labels, *args, **kargs):

#     logits = tf.cast(logits, tf.float32)
#     labels_ = tf.cast(labels, tf.int32)

#     labels = tf.cast(tf.expand_dims(labels_, -1), tf.int32)

#     predictions = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
#     batch_idxs = tf.range(0, tf.shape(labels)[0])
#     batch_idxs = tf.expand_dims(batch_idxs, 1)

#     idxs = tf.concat([batch_idxs, labels], 1)
#     y_true_pred = tf.gather_nd(predictions, idxs)

#     losses = -tf.reduce_mean(y_true_pred)
#     return losses, tf.exp(predictions)

def center_loss_v1(embedding, labels, *args, **kargs):
    '''
    embedding dim : (batch_size, num_features)
    '''
    config = args[0]
    num_features = embedding.get_shape()[-1]
    with tf.variable_scope(config.scope+"_center_loss"):
        centroids = tf.get_variable('center',
                        shape=[config.num_classes, num_features],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=False)

        centroids_delta = tf.get_variable('centroidsUpdateTempVariable',
                        shape=[config.num_classes, num_features],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                        trainable=False)

        centroids_batch = tf.gather(centroids, labels)
        # cLoss = tf.nn.l2_loss(embedding - centroids_batch) / (batch_size) # Eq. 2
        
        cLoss = tf.reduce_mean(tf.reduce_sum((embedding - centroids_batch)**2, axis=-1))


        diff = centroids_batch - embedding

        delta_c_nominator = tf.scatter_add(centroids_delta, labels, diff)
        indices = tf.expand_dims(labels, -1)
        updates = tf.cast(tf.ones_like(labels), tf.float32)
        shape = tf.constant([num_features])

        labels_sum = tf.expand_dims(tf.scatter_nd(indices, updates, shape),-1)
        centroids = centroids.assign_sub(config.alpha * delta_c_nominator / (1.0 + labels_sum))

        centroids_delta = centroids_delta.assign(tf.zeros([config.num_classes, num_features]))

        return cLoss, centroids

def center_loss_v2(features, labels, centers=None, *args, **kargs):
    config = kargs["config"]
    alpha = config.alpha
    num_classes = config.num_classes
    with tf.variable_scope(config.scope+"_center_loss"):
        len_features = features.get_shape()[1]
        if config.scope != "LabelNetwork":
            centers = tf.get_variable('centers', 
                            [num_classes, len_features], 
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            trainable=False)
     
        centers_batch = tf.gather(centers, labels)

        loss = tf.nn.l2_loss(features - centers_batch)
     
        diff = centers_batch - features
     
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
     
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers = tf.scatter_sub(centers, labels, diff)
     
        return loss, centers

def spread_loss(labels, activations, margin):
    activations_shape = activations.get_shape().as_list()
    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)    
    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )    
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )    
    gap_mit = tf.reduce_sum(tf.square(tf.nn.relu(margin - (activations_t - activations_i))))
    return gap_mit   

def margin_loss(y, preds):    
    y = tf.cast(y,tf.float32)
    loss = y * tf.square(tf.maximum(0., 0.9 - preds)) + \
        0.25 * (1.0 - y) * tf.square(tf.maximum(0., preds - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))  
    return loss


def multi_label_loss(logits, labels, *args, **kargs):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, 
                    labels=labels)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

def multi_label_hot(prediction, *args, **kargs):
    threshold = kargs["threshold"]
    prediction = tf.cast(prediction, tf.float32)
    threshold = float(threshold)
    pred_label = tf.cast(tf.geater(prediction, threshold), tf.int32)

    return pred_label

