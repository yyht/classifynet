import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.label_network import label_network_utils
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)



class LabelNetwork(ModelTemplate):
    def __init__(self):
        super(LabelNetwork, self).__init__()

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
            self.class_emb_mat = self.config.get("class_emb_mat", None)
            # ---- place holder -----
           
            self.sent_token = tf.placeholder(tf.int32, [None, None], name='sent_token')
            self.entity_token = tf.placeholder(tf.int32, [None, None], name='entity_token')
            self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')

            self.entity_token_mask = tf.cast(self.entity_token, tf.bool)
            self.entity_token_len = tf.reduce_sum(tf.cast(self.entity_token_mask, tf.int32), -1)

            self.sent_token_mask = tf.cast(self.sent_token, tf.bool)
            self.sent_token_len = tf.reduce_sum(tf.cast(self.sent_token_mask, tf.int32), -1)

            if self.config.with_char:

                self.sent_char = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
                self.sent_char_mask = tf.cast(self.sent_char, tf.bool)
                self.sent_char_len = tf.reduce_sum(tf.cast(self.sent_char_mask, tf.int32), -1)
                
                self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'gene_char_emb_mat')

            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'gene_token_emb_mat')
            with tf.variable_scope(self.scope + '_class_emb'):
                try:
                    print("===pre-trained class emb")
                    self.memory = tf.get_variable("label_memory", 
                                            shape=[self.num_classes, self.emb_size],
                                            dtype=tf.float32,
                                            initializer=tf.constant(self.class_emb_mat),
                                            trainable=True)

                except:
                    print("===random initialize class emb")
                    self.memory = tf.get_variable("label_memory", 
                                            shape=[self.num_classes, self.emb_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(-0.001, 0.001),
                                            trainable=True)
                
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
        entity_emb = tf.nn.embedding_lookup(self.emb_mat, self.entity_token)
        with tf.variable_scope(self.config.scope+"_entity_emb", reuse=None):
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

            # entity_emb = qanet_layers.conv(inputs=entity_emb,
            #                             output_size=self.emb_size,
            #                             kernel_size=1)
            
            # entity_mask = tf.expand_dims(self.entity_token_mask, axis=-1) # batch x len x 1
            # entity_emb = tf.reduce_max(qanet_layers.mask_logits(entity_emb, entity_mask), axis=1)
            # print(entity_emb.get_shape(), "=====emb shape=====")

            # entity_emb_ = tf.expand_dims(entity_emb, axis=1)

            entity_emb_ = tf.tile(entity_emb, [1, tf.shape(word_emb)[1], 1])

        mask = tf.expand_dims(self.sent_token_mask, -1)
        word_emb = tf.concat([word_emb, entity_emb_], axis=-1)
        word_emb *= tf.cast(mask, tf.float32)

        print(word_emb.get_shape(), "=====word with entity========")
        if self.config.with_char:
            char_emb = self.build_char_embedding(self.sent_char, self.sent_char_len, self.char_mat,
                    is_training=self.is_training, reuse=reuse)
            word_emb = tf.concat([word_emb, char_emb], axis=-1)
        
        return word_emb, entity_emb

    def build_encoder(self, input_lengths, input_mask, *args, **kargs):

        reuse = kargs["reuse"]
        word_emb, entity_emb = self.build_emebdding(*args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
            mask = tf.expand_dims(input_mask, -1)

            # sent_repres = tf.layers.dense(sent_repres, self.emb_size)

            sent_repres *= tf.cast(mask, tf.float32)
            # sent_repres = label_network_utils.self_attn(
            #     enc=sent_repres,
            #     scope=self.config.scope,
            #     dropout=dropout_rate,
            #     reuse=None,
            #     config=self.config
            #     )

            # sent_repres = label_network_utils.text_cnn(
            #         sent_repres,
            #         filter_sizes=[1,3,5],
            #         scope=self.config.scope,
            #         embed_size=self.emb_size,
            #         num_filters=self.config.num_filters)
            # output = sent_repres
            # print(sent_repres.get_shape(), "===text cnn encoder shape===")
            [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                            self.config.context_lstm_dim, 
                            input_lengths=input_lengths, 
                            scope_name=self.config.scope, 
                            reuse=reuse, 
                            is_training=self.is_training,
                            dropout_rate=dropout_rate, 
                            use_cudnn=self.config.use_cudnn)
            match_dim = self.config.context_lstm_dim * 8

        with tf.variable_scope(self.config.scope+"sent_label_attention", reuse=reuse):

            memory = tf.expand_dims(self.memory, axis=0)
            memory = tf.tile(memory, [tf.shape(sent_repres)[0],1,1])
            # entity_emb = tf.expand_dims(entity_emb, axis=1)
            # entity_emb = tf.tile(entity_emb, [1, tf.shape(memory)[1], 1])
            # print("===emb shape===", entity_emb.get_shape())
            # # batch x classes x dim
            # memory = tf.concat([memory, entity_emb], axis=-1)
            print("==memory shape==", memory.get_shape())

            # output = label_network_utils.memory_attention(sent_repres,
            #         memory, input_mask,
            #         scope=self.config.scope,
            #         memory_mask=None)
            print(sent_repres.get_shape(), memory.get_shape())
            output = label_network_utils.memory_attention_v1(
                    sent_repres, memory, 
                    input_mask, "memory_attention",
                    memory_mask=None,
                    reuse=None,
                    attention_output="multi_head",
                    num_heads=4,
                    dropout_rate=dropout_rate,
                    threshold=1/float(self.num_classes),
                    apply_hard_attn=True)
            print("==output shape==", output.get_shape())

            return sent_repres, entity_emb, output

    def build_discriminator(self, output, **kargs):
        reuse = kargs["reuse"]
        with tf.variable_scope(self.config.scope+"_discriminator", reuse=reuse):

            output = label_network_utils.multi_highway_layer(output, 
                                self.config.highway_layer_num)
            return output

    def build_predictor(self, output,
                        *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            
            matched_repres = tf.nn.dropout(output, (1 - dropout_rate))
            logits = tf.layers.dense(output, self.num_classes, use_bias=False)
            pred_probs = tf.nn.softmax(logits)

            return logits, pred_probs

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
        if self.config.with_center_loss:
            self.center_loss, _ = point_wise_loss.center_loss_v2(
                                            self.sent_repres, 
                                            self.gold_label, 
                                            centers=self.memory,
                                            config=self.config, 
                                            *args, **kargs)
            self.loss = self.loss + self.config.center_gamma * self.center_loss

        if self.config.get("mode", "train") == "train":
            if self.config.with_label_regularization:
                print("===with class regularization===")
                self.class_loss, _ = point_wise_loss.focal_loss_multi_v1(
                                        self.class_logits, self.gold_label, 
                                        self.config, *args, **kargs)
                self.loss += self.config.class_penalty * self.class_loss
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
        print("List of Variables:")
        for v in trainable_vars:
            print(v.name)

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        [_, 
        self.entity_emb, 
        self.sent_repres] = self.build_encoder(self.sent_token_len,
                                        self.sent_token_mask, 
                                        reuse = None)
        
        [self.logits, self.pred_probs] = self.build_predictor(self.sent_repres,
                            reuse=None)
        if self.config.get("mode", "train") == "train":

            memory_embed = tf.nn.embedding_lookup(self.memory,
                                            self.gold_label)
            
            [self.class_logits, self.class_pred_probs] = self.build_predictor(
                            memory_embed,
                            reuse=True)

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