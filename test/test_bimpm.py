import pickle as pkl
import tensorflow as tf
import time
import datetime
import numpy as np
import argparse

from random import random

import sys,os

sys.path.append("..")

from model.bimpm.bimpm import BiMPM
from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

# train_data_path = "/data/xuht/duplicate_sentence/ChineseSTSCorpus/train.txt"
train_data_path = "/data/xuht/duplicate_sentence/LCQMC/train.txt"
w2v_path = "/data/xuht/Chinese_w2v/sgns.merge.char/sgns.merge.char.pkl"
# vocab_path = "/data/xuht/duplicate_sentence/ChineseSTSCorpus/emb_mat.pkl"
vocab_path = "/data/xuht/duplicate_sentence/LCQMC/emb_mat.pkl"

data_clearner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

import time

[train_anchor, 
train_check, 
train_label, 
train_anchor_len, 
train_check_len] = data_utils.read_data(train_data_path, 
                    "train", 
                    cut_tool, 
                    data_clearner_api,
                    "tab")
                
dic = data_utils.make_dic(train_anchor+train_check)
data_utils.read_pretrained_embedding(w2v_path, dic, vocab_path, min_freq=3)

if sys.version_info < (3, ):
    embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
else:
    embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                            encoding="iso-8859-1")

token2id = embedding_info["token2id"]
id2token = embedding_info["id2token"]
embedding_mat = embedding_info["embedding_matrix"]
extral_symbol = embedding_info["extra_symbol"]

def train(FLAGS):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=False,
          log_device_placement=True,
          gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        print("started session")

        with sess.as_default():
            
            model = BiMPM()
            model.build_placeholder(FLAGS)
            model.build_op(is_training=True)
            model.init_step(sess)

            for epoch in range(FLAGS.max_epochs):

                train_data = get_batch_data.get_batches(train_anchor, 
                    train_check, 
                    train_label, FLAGS.batch_size, 
                    token2id, is_training=True)

                for corpus in train_data:
                    anchor, check, label = corpus
                    [loss, _, global_step, 
                        accuracy, preds] = model.step(sess, 
                                            [anchor, check, label])
                    print(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the train set.')
    args, unparsed = parser.parse_known_args()
    FLAGS = namespace_utils.load_namespace(args.config_path)

    FLAGS.token_emb_mat = embedding_mat
    FLAGS.char_emb_mat = 0
    FLAGS.vocab_size = embedding_mat.shape[0]
    FLAGS.char_vocab_size = 0
    FLAGS.emb_size = embedding_mat.shape[1]
    FLAGS.extra_symbol = extral_symbol

    train(FLAGS)




