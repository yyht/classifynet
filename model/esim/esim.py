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


EPSILON = 1e-8

class ESIM(ModelTemplate):
    def __init__(self):
        super(ESIM, self).__init__()
     
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
        # seq_len = tf.reduce_max(self.sent_token_len)
        # entity_emb = tf.tile(entity_emb, [1, seq_len, 1])

        # mask = tf.expand_dims(self.sent_token_mask, -1)
        # word_emb = tf.concat([word_emb, entity_emb], axis=-1)
        # word_emb *= tf.cast(mask, tf.float32)

        if self.config.with_target:
            entity_emb = tf.nn.embedding_lookup(self.emb_mat, self.entity_token)
            entity_emb_ = tf.expand_dims(entity_emb, axis=1)

            entity_emb_ = tf.tile(entity_emb_, [1, tf.shape(word_emb)[1], 1])
            mask = tf.expand_dims(self.sent_token_mask, -1)
            word_emb = tf.concat([word_emb, entity_emb_], axis=-1)
            word_emb *= tf.cast(mask, tf.float32)

        if self.config.with_idf:
            print("====apply idf embedding====")
            idf_emb = tf.nn.embedding_lookup(self.idf_mat,
                                            self.sent_token)

            word_emb = tf.concat([word_emb, idf_emb], axis=-1)
            mask = tf.expand_dims(self.sent_token_mask, -1)
            word_emb *= tf.cast(mask, tf.float32)

        print(word_emb.get_shape(), "=====word with entity========")
        if self.config.with_char:
            char_emb = self.build_char_embedding(self.sent_char, self.sent_char_len, self.char_mat,
                    is_training=self.is_training, reuse=reuse)
            word_emb = tf.concat([word_emb, char_emb], axis=-1)
        
        return word_emb
        
    def build_encoder(self, input_lengths, input_mask, *args, **kargs):

        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(*args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
            
            if self.config.rnn == "lstm":
                [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                                self.config.context_lstm_dim, 
                                input_lengths=input_lengths, 
                                scope_name=self.config.scope, 
                                reuse=reuse, 
                                is_training=self.is_training,
                                dropout_rate=dropout_rate, 
                                use_cudnn=self.config.use_cudnn)
                match_dim = self.config.context_lstm_dim * 6

            elif self.config.rnn == "slstm":

                word_emb_proj = tf.layers.dense(word_emb, 
                                        self.config.slstm_hidden_size)

                initial_hidden_states = word_emb_proj
                initial_cell_states = tf.identity(initial_hidden_states)

                [new_hidden_states, 
                new_cell_states, 
                dummynode_hidden_states] = slstm_utils.slstm_cell(self.config, 
                                    self.config.scope, 
                                    self.config.slstm_hidden_size, 
                                    input_lengths, 
                                    initial_hidden_states, 
                                    initial_cell_states, 
                                    self.config.slstm_layer_num,
                                    dropout_rate, reuse=reuse)

                sent_repres = new_hidden_states
                match_dim = self.config.slstm_hidden_size * 3

            if self.config.multi_head:
                mask = tf.cast(input_mask, tf.float32)
                ignore_padding = (1 - mask)
                ignore_padding = label_network_utils.attention_bias_ignore_padding(ignore_padding)
                encoder_self_attention_bias = ignore_padding

                sent_repres = label_network_utils.multihead_attention_texar(
                    sent_repres, 
                    memory=None, 
                    memory_attention_bias=encoder_self_attention_bias,
                    num_heads=8, 
                    num_units=128, 
                    dropout_rate=dropout_rate, 
                    scope="multihead_attention")

            v_attn = self_attn.multi_dimensional_attention(
                sent_repres, input_mask, 'multi_dim_attn_for_%s' % self.config.scope,
                1 - dropout_rate, self.is_training, self.config.weight_decay, "relu")
            
            mask = tf.expand_dims(input_mask, -1)
            v_sum = tf.reduce_sum(sent_repres*tf.cast(mask, tf.float32), 1)
            v_ave = tf.div(v_sum, tf.expand_dims(tf.cast(input_lengths, tf.float32)+EPSILON, -1))

            v_max = tf.reduce_max(qanet_layers.mask_logits(sent_repres, mask), axis=1)

            v_last = esim_utils.last_relevant_output(sent_repres, input_lengths)

            out = tf.concat([v_ave, v_max, v_last, v_attn], axis=-1)

        return out, match_dim

    def build_predictor(self, matched_repres, *args, **kargs):
        match_dim = kargs["match_dim"]
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim

            matched_repres = tf.nn.dropout(matched_repres, (1 - dropout_rate))
            self.logits = tf.layers.dense(matched_repres, num_classes, use_bias=False)
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
            self.center_loss, _ = point_wise_loss.center_loss_v2(
                                            features=self.sent_repres, 
                                            labels=self.gold_label,  
                                            config=self.config)
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

        self.sent_repres, match_dim = self.build_encoder(self.sent_token_len,
                                        self.sent_token_mask, 
                                        reuse = None)
        if self.config.get("l2_normalization", False):
            print("=======apply normalization=======")
            self.sent_repres = tf.nn.l2_normalize(self.sent_repres, axis=-1)

        self.build_predictor(self.sent_repres,
                            reuse = None,
                            match_dim = match_dim)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent_token, entity_token, gold_label] = sample_batch
        learning_rate = kargs["learning_rate"]
        feed_dict = {
            self.sent_token: sent_token,
            self.entity_token: entity_token,
            self.gold_label: gold_label,
            self.learning_rate: learning_rate,
            self.is_training: kargs["is_training"]
        }
        return feed_dict