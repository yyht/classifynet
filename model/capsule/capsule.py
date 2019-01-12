import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

from model.utils.capsule import capsule_modules
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os

EPSILON = 1e-8

class Capsule(ModelTemplate):
    def __init__(self):
        super(Capsule, self).__init__()

    def build_placeholder(self, config):
        self.config = config
        with self.graph.as_default():
            self.token_emb_mat = self.config["token_emb_mat"]
            self.char_emb_mat = self.config["char_emb_mat"]
            self.vocab_size = int(self.config["vocab_size"])
            self.char_vocab_size = int(self.config["char_vocab_size"])
            self.max_length = int(self.config["max_length"])
            self.emb_size = int(self.config["emb_size"])
            self.extra_symbol = self.config["extra_symbol"]
            self.scope = self.config["scope"]
            self.num_classes = int(self.config["num_classes"])
            self.batch_size = int(self.config["batch_size"])
            self.grad_clipper = float(self.config.get("grad_clipper", 10.0))
            self.char_limit = self.config.get("char_limit", 10)
            self.char_dim = self.config.get("char_emb_size", 300)
            self.entity_max_len = self.config.get("entity_max_len", 20)
            # ---- place holder -----
            if self.config.is_train:
                self.sent_token = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='sent_token')
                self.entity_token = tf.placeholder(tf.int32, [self.batch_size, self.entity_max_len], name='entity_token')
                self.gold_label = tf.placeholder(tf.int32, [self.batch_size], name='gold_label')
            else:
                self.sent_token = tf.placeholder(tf.int32, [1, self.max_length], name='sent_token')
                self.entity_token = tf.placeholder(tf.int32, [1, self.entity_max_len], name='entity_token')
                self.gold_label = tf.placeholder(tf.int32, [1], name='gold_label')

            self.entity_token_mask = tf.cast(self.entity_token, tf.bool)
            self.entity_token_len = tf.reduce_sum(tf.cast(self.entity_token_mask, tf.int32), -1)

            self.sent_token_mask = tf.cast(self.sent_token, tf.bool)
            self.sent_token_len = tf.reduce_sum(tf.cast(self.sent_token_mask, tf.int32), -1)

            if self.config.with_char:
                # self.sent1_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
                # self.sent2_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
                if self.config.is_train:
                    self.sent_char = tf.placeholder(tf.int32, [self.batch_size, self.max_length, self.char_limit]) # [batch_size, question_len, q_char_len]
                else:
                    self.sent_char = tf.placeholder(tf.int32, [1, self.max_length, self.char_limit]) # [batch_size, question_len, q_char_len]
                self.sent_char_mask = tf.cast(self.sent_char, tf.bool)
                self.sent_char_len = tf.reduce_sum(tf.cast(self.sent_char_mask, tf.int32), -1)
                
                self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope='gene_char_emb_mat')

            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope='gene_token_emb_mat')

            

            # ---------------- for dynamic learning rate -------------------
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            

    def build_char_embedding(self, char_token, char_lengths, char_embedding, *args, **kargs):

        reuse = kargs["reuse"]
        if self.config.char_embedding == "lstm":
            char_emb = char_embedding_utils.lstm_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        elif self.config.char_embedding == "conv":
            char_emb = char_embedding_utils.conv_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        return char_emb

    def build_emebdding(self, *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent_token)
        if self.config.with_target:
            entity_emb = tf.nn.embedding_lookup(self.emb_mat, self.entity_token)
            entity_emb_ = tf.expand_dims(entity_emb, axis=1)

            entity_emb_ = tf.tile(entity_emb_, [1, tf.shape(word_emb)[1], 1])
            mask = tf.expand_dims(self.sent_token_mask, -1)
            word_emb = tf.concat([word_emb, entity_emb_], axis=-1)
            word_emb *= tf.cast(mask, tf.float32)
        # entity_emb = tf.nn.embedding_lookup(self.emb_mat, self.entity_token)

        # [_, 
        # _, 
        # entity_emb] = layer_utils.my_lstm_layer(entity_emb, 
        #                         self.config.context_lstm_dim, 
        #                         input_lengths=self.entity_token_len, 
        #                         scope_name=self.config.scope, 
        #                         reuse=reuse, 
        #                         is_training=self.is_training,
        #                         dropout_rate=dropout_rate, 
        #                         use_cudnn=self.config.use_cudnn)

        # entity_mask = tf.expand_dims(self.entity_token_mask, axis=-1) # batch x len x 1
        # entity_emb = tf.reduce_max(qanet_layers.mask_logits(entity_emb, entity_mask), axis=1)
        
        # entity_emb = tf.expand_dims(entity_emb, axis=1)
        # entity_emb = tf.tile(entity_emb, [1, self.max_length, 1])

        # mask = tf.expand_dims(self.sent_token_mask, -1)
        # word_emb = tf.concat([word_emb, entity_emb], axis=-1)
        # word_emb *= tf.cast(mask, tf.float32)

        print(word_emb.get_shape(), "=====word with entity========")
        if self.config.with_char:
            char_emb = self.build_char_embedding(self.sent_char, self.sent_char_len, self.char_mat,
                    is_training=self.is_training, reuse=reuse)
            word_emb = tf.concat([word_emb, char_emb], axis=-1)
        
        return word_emb

    def build_encoder(self, *args, **kargs):
        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(*args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]

            sent_repres = tf.layers.dense(word_emb, self.emb_size, 
                activation=tf.nn.relu) + word_emb

            # sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)

            sent_repres = tf.expand_dims(sent_repres, axis=-1)
            print("===sent repres shape===", sent_repres.get_shape())
            if self.config.get("capsule_type", "capsule_a") == "capsule_a":
                [poses, activations] = capsule_modules.capsule_model_A(sent_repres, 
                                            self.config.num_classes)
            elif self.config.get("capsule_type", "capsule_b") == "capsule_b":
                [poses, activations] = capsule_modules.capsule_model_B(sent_repres, 
                                            self.config.num_classes)
            elif self.config.get("capsule_type", "CNN") == "CNN":
                [poses, activations] = capsule_modules.capsule_model_B(sent_repres, 
                                            self.config.num_classes)
            
            return [poses, activations]

    def build_predictor(self, poses, activations, 
                        *args, **kargs):

        self.logits = activations
        self.pred_probs = tf.nn.softmax(activations)
        
    def build_loss(self, *args, **kargs):
        if self.config.loss == "softmax_loss":
            self.loss, _ = point_wise_loss.softmax_loss(self.logits, self.gold_label, 
                                    *args, **kargs)
        elif self.config.loss == "sparse_amsoftmax_loss":
            self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_multi_v1":
            self.loss, _ = point_wise_loss.focal_loss_multi_v1(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "spread_loss":
            self.loss = point_wise_loss.spread_loss(self.logits, self.gold_label, 
                                        self.config.margin)
        elif self.config.loss == "margin_loss":
            self.loss = point_wise_loss.margin_loss(self.logits, 
                                        self.gold_label)

        if self.config.with_center_loss:
            self.center_loss, _ = point_wise_loss.center_loss_v2(self.sent_repres, 
                                            self.gold_label, self.config, 
                                            *args, **kargs)
            self.loss = self.loss + self.config.center_gamma * self.center_loss

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        [self.poses, self.activations] = self.build_encoder(self.sent_token_len,
                                        self.sent_token_mask, 
                                        reuse = None)
        self.build_predictor(self.poses, self.activations)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent_token, entity_token, gold_label] = sample_batch

        feed_dict = {
            self.sent_token: sent_token,
            self.entity_token: entity_token,
            self.gold_label: gold_label,
            self.learning_rate: self.config.learning_rate,
            self.is_training: kargs["is_training"]
        }
        return feed_dict