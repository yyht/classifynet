import tensorflow as tf
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.transformer import base_transformer_utils
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

class BaseTransformer(ModelTemplate):
    def __init__(self):
        super(BaseTransformer, self).__init__()

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
        if self.config.with_char:
            char_emb = self.build_char_embedding(self.sent_char, self.sent_char_len, self.char_mat,
                    is_training=is_training, reuse=reuse)
            word_emb = tf.concat([word_emb, char_emb], axis=-1)
        
        return word_emb

    def build_encoder(self, input_mask, *args, **kargs):
        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(*args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        input_mask = tf.cast(input_mask, tf.float32)
        input_mask = tf.expand_dims(input_mask, axis=-1) # batch_size x seq_len x 1
        word_emb *= input_mask

        input_length = tf.reduce_sum(input_mask, axis=-1)

        word_emb = tf.layers.dense(word_emb, self.config.hidden_size)
    
        with tf.variable_scope(self.config.scope+"_transformer_encoder", 
                    reuse=reuse):
            encoder_output = base_transformer_utils.transformer_encoder(word_emb, 
                                        target_space=None, 
                                        hparams=self.config, 
                                        features=None, 
                                        losses=None)

            input_mask = tf.squeeze(input_mask, axis=-1)
            v_attn = self_attn.multi_dimensional_attention(
                encoder_output, input_mask, 'multi_dim_attn_for_%s' % self.config.scope,
                1 - dropout_rate, self.is_training, self.config.weight_decay, "relu")
            
            v_sum = tf.reduce_sum(encoder_output, 1)
            v_ave = tf.div(v_sum, tf.expand_dims(tf.cast(input_length, tf.float32), -1))
            v_max = tf.reduce_max(encoder_output, 1)

            out = tf.concat([v_ave, v_max, v_attn], axis=-1)
            return out

    def build_predictor(self, matched_repres, *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        self.logits = nn.linear([matched_repres], 
                            self.config.num_classes, 
                            True, 0., scope= self.scope+'_logits', 
                            squeeze=False,
                            wd=self.config.weight_decay, 
                            input_keep_prob=1 - dropout_rate,
                            is_train=self.is_training)

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
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent_repres = self.build_encoder(self.sent_token_mask, 
                                        reuse = None)

        self.build_predictor(self.sent_repres,
                            reuse = None)

        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent_token, gold_label] = sample_batch

        feed_dict = {
            self.sent_token: sent_token,
            self.gold_label:gold_label,
            self.learning_rate: self.config.learning_rate,
            self.is_training:kargs["is_training"]
        }
        return feed_dict