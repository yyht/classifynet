import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os

class ModelTemplate(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)

        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          gpu_options=gpu_options)
        self.sess = tf.Session(config=session_conf,
                                graph=self.graph)

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

            self.idf_emb_mat = self.config.get("idf_emb_mat", None)

            # ---- place holder -----
            self.sent_token = tf.placeholder(tf.int32, [None, None], name='sent_token')
            self.entity_token = tf.placeholder(tf.int32, [None, None], name='entity_token')
            self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
            
            self.entity_token_mask = tf.cast(self.entity_token, tf.bool)
            self.entity_token_len = tf.reduce_sum(tf.cast(self.entity_token_mask, tf.int32), -1)

            self.sent_token_mask = tf.cast(self.sent_token, tf.bool)
            self.sent_token_len = tf.reduce_sum(tf.cast(self.sent_token_mask, tf.int32), -1)

            if self.config.with_char:
                # self.sent1_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
                # self.sent2_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
                self.sent_char = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]

                self.sent_char_mask = tf.cast(self.sent_char, tf.bool)
                self.sent_char_len = tf.reduce_sum(tf.cast(self.sent_char_mask, tf.int32), -1)
                
                self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_char_emb_mat')
            
            if self.config.with_idf:
                print("====apply idf embedding====")
                self.idf_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=1,
                                     init_mat=self.idf_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_idf_emb_mat')


            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_token_emb_mat')

            

            # ---------------- for dynamic learning rate -------------------
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            
    @abstractmethod
    def build_embedding(self, *args, **kargs):
        pass

    @abstractmethod
    def build_char_embedding(self, *args, **kargs):
        pass

    @abstractmethod
    def build_model(self, *args, **kargs):
        pass

    @abstractmethod
    def build_encoder(self, *args, **kargs):
        pass

    @abstractmethod
    def build_interactor(self, *args, **kargs):
        pass

    @abstractmethod
    def build_predictor(self, *args, **kargs):
        pass

    @abstractmethod
    def build_loss(self, *args, **kargs):
        pass

    @abstractmethod
    def build_accuracy(self, *args, **kargs):
        pass

    def apply_ema(self, *args, **kargs):
        decay = self.config.get("with_moving_average", None)
        if decay:
            with self.graph.as_default():
                self.var_ema = tf.train.ExponentialMovingAverage(decay)
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.loss = tf.identity(self.loss)

                    self.shadow_vars = []
                    self.global_vars = []
                    for var in tf.global_variables():
                        v = self.var_ema.average(var)
                        if v:
                            self.shadow_vars.append(v)
                            self.global_vars.append(var)
                    self.assign_vars = []
                    for g,v in zip(self.global_vars, self.shadow_vars):
                        self.assign_vars.append(tf.assign(g,v))

    def build_op(self, *args, **kargs):

        with self.graph.as_default():
        
            self.build_model(*args, **kargs)
            self.build_loss(*args, **kargs)
            self.build_accuracy(*args, **kargs)

            self.apply_ema(*args, **kargs)

            # ---------- optimization ---------
            if self.config["optimizer"].lower() == 'adadelta':
                self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.config["optimizer"].lower() == 'adam':
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
            elif self.config["optimizer"].lower() == 'rmsprop':
                self.opt = tf.train.RMSPropOptimizer(self.learning_rate)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
            grads_and_vars = self.opt.compute_gradients(self.loss, var_list=trainable_vars)
            
            params = [var for _, var in grads_and_vars]
            gradients = [grad for grad, _ in grads_and_vars]

            grads, _ = tf.clip_by_global_norm(gradients, self.grad_clipper)

            # grads = [tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad) for grad in grads]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
            with tf.control_dependencies(update_ops):  #ADD 2018.06.01
                self.train_op = self.opt.apply_gradients(zip(grads, params), global_step=self.global_step)
            self.saver = tf.train.Saver(max_to_keep=100)

            self.grad_vars = [grad for grad, _ in grads_and_vars]

    def init_step(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def save_model(self, model_dir, model_str):
        with self.graph.as_default():
            self.saver.save(self.sess, 
                    os.path.join(model_dir, 
                                model_str+".ckpt"))

    def load_model(self, model_dir, model_str):
        with self.graph.as_default():
            model_path = os.path.join(model_dir, model_str+".ckpt")
            self.saver.restore(self.sess, model_path)
            print("===model path===", model_path)
            if self.config.get("with_moving_average", None):
                self.sess.run(self.assign_vars)
        
    def step(self, batch_samples, *args, **kargs):
        feed_dict = self.get_feed_dict(batch_samples, *args, **kargs)
        with self.graph.as_default():
            [loss, train_op, global_step, 
            accuracy, preds] = self.sess.run([self.loss, self.train_op, 
                                          self.global_step, 
                                          self.accuracy, 
                                          self.pred_probs],
                                          feed_dict=feed_dict)
        return [loss, train_op, global_step, 
                    accuracy,preds]

    def infer(self, batch_samples, mode, *args, **kargs):
        feed_dict = self.get_feed_dict(batch_samples, *args, **kargs)
        if mode == "test":
            with self.graph.as_default():
                [loss, logits, pred_probs, accuracy] = self.sess.run([self.loss, self.logits, 
                                                            self.pred_probs, 
                                                            self.accuracy], 
                                                            feed_dict=feed_dict)
            return loss, logits, pred_probs, accuracy
        elif mode == "infer":
            with self.graph.as_default():
                [logits, pred_probs, sent_repres] = self.sess.run([self.logits, 
                                                self.pred_probs,
                                                self.sent_repres], 
                                            feed_dict=feed_dict)
            return logits, pred_probs, sent_repres
