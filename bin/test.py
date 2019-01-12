import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse

from random import random

import sys,os

sys.path.append("..")

from model.bimpm.bimpm import BiMPM
from model.esim.esim import ESIM
from model.biblosa.biblosa import BiBLOSA
from model.transformer.base_transformer import BaseTransformer
from model.transformer.universal_transformer import UniversalTransformer

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_clearner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

def prepare_data(data_path, w2v_path, vocab_path, make_vocab=True):

    [anchor, 
    check, 
    label, 
    anchor_len, 
    check_len] = data_utils.read_data(data_path, 
                    "train", 
                    cut_tool, 
                    data_clearner_api,
                    "tab")

    if make_vocab:
        dic = data_utils.make_dic(anchor+check)
        data_utils.read_pretrained_embedding(w2v_path, dic, vocab_path, min_freq=3)

    if sys.version_info < (3, ):
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
    else:
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                encoding="iso-8859-1")

    return [anchor, check, label, anchor_len, check_len, embedding_info]

def test(config):
    model_config_path = config["model_config_path"]
    FLAGS = namespace_utils.load_namespace(model_config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu_id", "")
    w2v_path = config["w2v_path"]
    vocab_path = config["vocab_path"]
    test_path = config["test_path"]

    model_dir = config["model_dir"]
    model_str = config["model_str"]
    model_name = config["model"]

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.mkdir(os.path.join(model_dir, model_name))

    if not os.path.exists(os.path.join(model_dir, model_name, "logs")):
        os.mkdir(os.path.join(model_dir, model_name, "logs"))

    if not os.path.exists(os.path.join(model_dir, model_name, "models")):
        os.mkdir(os.path.join(model_dir, model_name, "models"))

    [test_anchor, 
    test_check, 
    test_label, 
    test_anchor_len, 
    test_check_len, 
    embedding_info] = prepare_data(test_path, 
                        w2v_path, vocab_path,
                        make_vocab=False)

    token2id = embedding_info["token2id"]
    id2token = embedding_info["id2token"]
    embedding_mat = embedding_info["embedding_matrix"]
    extral_symbol = embedding_info["extra_symbol"]

    FLAGS.token_emb_mat = embedding_mat
    FLAGS.char_emb_mat = 0
    FLAGS.vocab_size = embedding_mat.shape[0]
    FLAGS.char_vocab_size = 0
    FLAGS.emb_size = embedding_mat.shape[1]
    FLAGS.extra_symbol = extral_symbol

    if FLAGS.scope == "BiMPM":
        model = BiMPM()
    elif FLAGS.scope == "ESIM":
        model = ESIM()
    elif FLAGS.scope == "BiBLOSA":
        model = BiBLOSA()
    elif FLAGS.scope == "BaseTransformer":
        model = BaseTransformer()
    elif FLAGS.scope == "UniversalTransformer":
        model = UniversalTransformer()

    model.build_placeholder(FLAGS)
    model.build_op()
    model.init_step()
    model.load_model(os.path.join(model_dir, model_name, "models"), 
                    model_str)

    test_data = get_batch_data.get_batches(test_anchor, 
            test_check, 
            test_label, FLAGS.batch_size, 
            token2id, is_training=False)

    test_loss, test_accuracy = 0, 0
    cnt = 0
    for index, corpus in enumerate(test_data):
        anchor, check, label = corpus
        try:
            [loss, logits, 
                pred_probs, accuracy] = model.infer(
                                [anchor, check, label], 
                                mode="test",
                                is_training=False)

            test_loss += loss*anchor.shape[0]
            test_accuracy += accuracy*anchor.shape[0]
            cnt += anchor.shape[0]
            
        except:
            continue
       
    test_loss /= float(cnt)
    test_accuracy /= float(cnt)

    print(test_loss, test_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_config', type=str, help='model config path')
    parser.add_argument('--model_dir', type=str, help='model path')
    parser.add_argument('--config_prefix', type=str, help='config path')
    parser.add_argument('--gpu_id', type=str, help='gpu id')
    parser.add_argument('--test_path', type=str, help='train data path')
    parser.add_argument('--w2v_path', type=str, help='pretrained w2v path')
    parser.add_argument('--vocab_path', type=str, help='vocab_path')
    parser.add_argument('--model_str', type=str, help='vocab_path')

    args, unparsed = parser.parse_known_args()
    model_config = args.model_config

    with open(model_config, "r") as frobj:
        model_config = json.load(frobj)

    config = {}
    config["model_dir"] = args.model_dir
    config["model"] = args.model
    config["model_config_path"] = os.path.join(args.config_prefix, 
                            model_config.get(args.model, "biblosa"))
    config["gpu_id"] = args.gpu_id
    config["test_path"] = args.test_path
    config["w2v_path"] = args.w2v_path
    config["vocab_path"] = args.vocab_path
    config["model_str"] = args.model_str
    
    test(config)


    










