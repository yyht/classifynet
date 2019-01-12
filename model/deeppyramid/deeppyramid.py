import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.deeppyramid import deeppyramid_utils
from model.utils.embed import integration_func

EPSILON = 1e-8

class DeepPyramid(ModelTemplate):
    def __init__(self):
        super(DeepPyramid, self).__init__()

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
                self.entity_token = tf.placeholder(tf.int32, [self.batch_size, 1], name='entity_token')
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
                                     scope=self.scope+'_gene_char_emb_mat')

            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_token_emb_mat')

            

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
        seq_len = tf.shape(word_emb)[1]
        entity_emb = tf.tile(entity_emb, [1, seq_len, 1])

        mask = tf.expand_dims(self.sent_token_mask, -1)
        word_emb = tf.concat([word_emb, entity_emb], axis=-1)
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
            sent_repres = deeppyramid_utils.highway_network(word_emb, self.config.highway_layer_num, 
                                    True, wd=False, is_train=self.is_training,
                                    input_keep_prob=1.0 - dropout_rate)    
            
            sent_repres = tf.layers.dense(sent_repres, 100, activation=tf.nn.relu, use_bias=False)
            # mask = tf.expand_dims(input_mask, -1)
            # mask = tf.cast(mask, tf.float32)

            # sent_repres = qanet_layers.mask_logits(sent_repres, mask)

            output = deeppyramid_utils.deep_pyramid_cnn(self.config, sent_repres,
                                                self.is_training)
            output = tf.layers.dense(output, self.config.hidden_size*4, 
                            activation=tf.nn.relu, use_bias=True) #[batch_size,h*4]
            output = tf.nn.dropout(output, keep_prob=1 - dropout_rate) #[batch_size,h*4]
        return output

    def build_predictor(self, matched_repres, *args, **kargs):
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim

            matched_repres = tf.layers.dense(matched_repres, self.config.hidden_size, 
                            activation=tf.nn.relu, use_bias=True) #[batch_size,h*4]
            matched_repres = tf.nn.dropout(matched_repres, keep_prob=1 - dropout_rate) #[batch_size,h*4]
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
        learning_rate = kargs["learning_rate"]
        feed_dict = {
            self.sent_token: sent_token,
            self.entity_token: entity_token,
            self.gold_label: gold_label,
            self.learning_rate: learning_rate,
            self.is_training: kargs["is_training"]
        }
        return feed_dict