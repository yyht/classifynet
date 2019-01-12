import tensorflow as tf
import numpy as np
from model.utils import qanet_layers
from third_party.tensor2tensor.layers import common_attention, common_layers

def query_context_attention(query, context, max_query_len, max_context_len, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):
        context_ = tf.transpose(context, [0,2,1])
        S = tf.matmul(query, context_)  # batch x q_len x c_len

        mask_q = tf.expand_dims(query_mask, 1)
        mask_c = tf.expand_dims(context_mask, 1)

        S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
        c2q = tf.matmul(S_, context)

        S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        q2c = tf.matmul(S_T, query)

        query_attention_outputs = tf.concat([query, c2q, query-c2q, query*c2q], axis=-1)
        query_attention_outputs *= tf.expand_dims(query_mask, -1)

        context_attention_outputs = tf.concat([context, q2c, context-q2c, context*q2c], axis=-1)
        context_attention_outputs *= tf.expand_dims(query_mask, -1)

        return query_attention_outputs, context_attention_outputs