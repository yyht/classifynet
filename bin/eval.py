import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse

from random import random

import sys,os

sys.path.append("..")

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

data_cleaner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Eval(object):
    def __init__(self, config):
        self.config = config

        with open(self.config["model_config"], "r") as frobj:
            self.model_dict = json.load(frobj)

        self.model_config_path = self.config["model_config_path"]
        self.vocab_path = self.config["vocab_path"]

        if sys.version_info < (3, ):
            self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"))
        else:
            self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"), 
                                    encoding="iso-8859-1")

        self.token2id = self.embedding_info["token2id"]
        self.id2token = self.embedding_info["id2token"]
        self.embedding_mat = self.embedding_info["embedding_matrix"]
        self.extral_symbol = self.embedding_info["extra_symbol"]

    def init_model(self, model_config):

        model_name = model_config["model_name"]
        model_str = model_config["model_str"]
        model_dir = model_config["model_dir"]

        FLAGS = namespace_utils.load_namespace(os.path.join(self.model_config_path, model_name+".json"))
        if FLAGS.scope == "ESIM":
            model = ESIM()
        elif FLAGS.scope == "BiBLOSA":
            model = BiBLOSA()
        elif FLAGS.scope == "BaseTransformer":
            model = BaseTransformer()
        elif FLAGS.scope == "UniversalTransformer":
            model = UniversalTransformer()

        FLAGS.token_emb_mat = self.embedding_mat
        FLAGS.char_emb_mat = 0
        FLAGS.vocab_size = self.embedding_mat.shape[0]
        FLAGS.char_vocab_size = 0
        FLAGS.emb_size = self.embedding_mat.shape[1]
        FLAGS.extra_symbol = self.extral_symbol

        model.build_placeholder(FLAGS)
        model.build_op()
        model.init_step()
        model.load_model(model_dir, model_str)

        return model

    def init(self, model_config_lst):
        self.model = {}
        for model_name in model_config_lst:
            if model_name in self.model_dict:
                self.model[model_name] = self.init_model(model_config_lst[model_name])

    def prepare_data(self, question_lst):
        question_lst = [cut_tool.cut(data_cleaner_api.clean(question)) for question in question_lst]
        return question_lst

    def model_eval(self, model_name, question_lst):
        
        eval_batch = get_batch_data.get_eval_classify_batches(question_lst, 
                                    100, 
                                    self.token2id, 
                                    is_training=False)

        eval_probs = []
        sent_repres = []
        for batch in eval_batch:
            [logits, preds, repres] = self.model[model_name].infer(batch, mode="infer", is_training=False)
            eval_probs.extend(list(preds[:,1]))
            sent_repres.extend(repres)
        return eval_probs, sent_repres

    def infer(self, question_lst):
        question_lst = self.prepare_data(question_lst)
        eval_probs, sent_repres = {}, {}
        for model_name in self.model:
            probs, repres = self.model_eval(model_name, question_lst)
            eval_probs[model_name] = probs
            sent_repres[model_name] = repres
        return eval_probs, sent_repres

if __name__ == "__main__":
    config = {}
    config["model_config"] = "/notebooks/source/classifynet/model_config.json"
    config["model_config_path"] = "/notebooks/source/classifynet/configs"
    config["vocab_path"] = "/data/xuht/ChineseSTSListCorpus/emb_mat.pkl"

    model_config_lst = {}
    model_config_lst["esim"] = {
        "model_name":"esim",
        "model_str":"esim_1535445113_1.2367421343323532_0.8392472990259687",
        "model_dir":"/data/xuht/test/classifynet/esim/models"
    }
    
    eval_api = Eval(config)
    eval_api.init(model_config_lst)

    preds, sent_repres = eval_api.infer([    
                "广州客运站的数目",
                "广州有多少个客运站",
                "广州有几个汽车客运站",
                "广州天河有几个客运站",
                "广州天河区有几个汽车客运站",
                "深圳有几个客运站"])


    import pickle as pkl
    pkl.dump(sent_repres, 
            open("/data/xuht/test/classifynet/esim/test.pkl", "wb"))