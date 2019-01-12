import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.common import common_utils
from model.utils.swem import swem_utils
EPSILON = 1e-8

class SWEM(ModelTemplate):
    def __init__(self):
        super(SWEM, self).__init__()

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

        if self.config.with_word_drop:

            word_drop_rate = tf.cond(self.is_training, 
                            lambda:self.config.word_drop_rate,
                            lambda:0.0)
            word_emb, word_drop_mask = common_utils.word_dropout(word_emb, word_drop_rate)
        else:
            word_drop_mask = self.sent_token_mask
        entity_emb = tf.nn.embedding_lookup(self.emb_mat, self.entity_token)

        [_, 
        _, 
        entity_emb] = layer_utils.my_lstm_layer(entity_emb, 
                                self.config.context_lstm_dim, 
                                input_lengths=self.entity_token_len, 
                                scope_name=self.config.scope, 
                                reuse=reuse, 
                                is_training=self.is_training,
                                dropout_rate=dropout_rate, 
                                use_cudnn=self.config.use_cudnn)

        entity_mask = tf.expand_dims(self.entity_token_mask, axis=-1) # batch x len x 1
        entity_emb = tf.reduce_max(qanet_layers.mask_logits(entity_emb, entity_mask), axis=1)
        
        entity_emb = tf.expand_dims(entity_emb, axis=1)
        seq_len = tf.reduce_max(self.sent_token_len)
        entity_emb = tf.tile(entity_emb, [1, seq_len, 1])

        mask = tf.expand_dims(self.sent_token_mask, -1)
        word_emb = tf.concat([word_emb, entity_emb], axis=-1)
        word_emb *= tf.cast(mask, tf.float32)

        print(word_emb.get_shape(), "=====word with entity========")
        if self.config.with_char:
            char_emb = self.build_char_embedding(self.sent_char, self.sent_char_len, self.char_mat,
                    is_training=self.is_training, reuse=reuse)
            word_emb = tf.concat([word_emb, char_emb], axis=-1)
        
        return word_emb, word_drop_mask

    def build_encoder(self, input_lengths, input_mask, *args, **kargs):
        reuse = kargs["reuse"]
        word_emb, word_drop_mask = self.build_emebdding(*args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_drop_mask = tf.cast(word_drop_mask, tf.float32)
        word_drop_mask = tf.squeeze(word_drop_mask, axis=-1)
        
        input_mask = tf.cast(input_mask, tf.float32)
        input_mask *= word_drop_mask

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)

        H_enc_2 = tf.reduce_max(qanet_layers.mask_logits(word_emb, tf.expand_dims(input_mask, -1)), axis=1)
        
        input_mask = tf.expand_dims(input_mask, -1)
        H_enc_1 = tf.reduce_sum(word_emb*tf.cast(input_mask, tf.float32), 1)
        H_enc_1 = tf.div(H_enc_1, tf.reduce_sum(input_mask, axis=1)+EPSILON)

        H_enc = tf.concat([H_enc_1, H_enc_2], 1)

        return H_enc

    def build_predictor(self, matched_repres, *args, **kargs):

        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim

            self.logits = swem_utils.discriminator_2layer(matched_repres, self.config, 
                            1 - dropout_rate, prefix='classify_', 
                            num_outputs=self.config.num_classes, 
                            is_reuse=None)

            self.pred_probs = tf.nn.softmax(self.logits)

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

        self.sent_repres = self.build_encoder(self.sent_token_len,
                                        self.sent_token_mask, 
                                        reuse = None)
        if self.config.get("l2_normalization", False):
            print("=======apply normalization=======")
            self.sent_repres = tf.nn.l2_normalize(self.sent_repres, axis=-1)

        self.build_predictor(self.sent_repres,
                            reuse = None)

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