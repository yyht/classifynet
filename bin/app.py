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
from model.re_augument.re_augument import ReAugument

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_cleaner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()
cut_tool.init_config({
            "user_dict":"/data/xuht/eventy_detection/inference_data/project_entity.txt"})
cut_tool.build_tool()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Eval(object):
    def __init__(self, config):
        self.config = config

        with open(self.config["model_config"], "r") as frobj:
            self.model_dict = json.load(frobj)

        # self.model_config_path = self.config["model_config_path"]
        # self.vocab_path = self.config["vocab_path"]

        # if sys.version_info < (3, ):
        #     self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"))
        # else:
        #     self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"), 
        #                             encoding="iso-8859-1")

        # self.token2id = self.embedding_info["token2id"]
        # self.id2token = self.embedding_info["id2token"]
        # self.embedding_mat = self.embedding_info["embedding_matrix"]
        # self.extral_symbol = self.embedding_info["extra_symbol"]

    def init_model(self, model_config):

        model_name = model_config["model_name"]
        model_str = model_config["model_str"]
        model_dir = model_config["model_dir"]

        model_config_path = model_config["model_config_path"]
        vocab_path = model_config["vocab_path"]

        if sys.version_info < (3, ):
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
        else:
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                    encoding="iso-8859-1")

        if model_config.get("label_emb_path", None):
            label_emb_mat = pkl.load(open(label_emb_path, "rb"))
        else:
            label_emb_mat = None

        token2id = embedding_info["token2id"]
        id2token = embedding_info["id2token"]
        embedding_mat = embedding_info["embedding_matrix"]
        extral_symbol = embedding_info["extra_symbol"]

        FLAGS = namespace_utils.load_namespace(os.path.join(model_config_path, model_name+".json"))
        if FLAGS.scope == "ESIM":
            model = ESIM()
        elif FLAGS.scope == "BiBLOSA":
            model = BiBLOSA()
        elif FLAGS.scope == "BaseTransformer":
            model = BaseTransformer()
        elif FLAGS.scope == "UniversalTransformer":
            model = UniversalTransformer()
        elif FLAGS.scope == "ReAugument":
            model = ReAugument()

        FLAGS.token_emb_mat = embedding_mat
        FLAGS.char_emb_mat = 0
        FLAGS.vocab_size = embedding_mat.shape[0]
        FLAGS.char_vocab_size = 0
        FLAGS.emb_size = embedding_mat.shape[1]
        FLAGS.extra_symbol = extral_symbol
        FLAGS.class_emb_mat = label_emb_mat

        model.build_placeholder(FLAGS)
        model.build_op()
        model.init_step()
        model.load_model(model_dir, model_str)

        return model, token2id

    def init(self, model_config_lst):
        self.model = {}
        self.token2id = {}
        for model_name in model_config_lst:
            print(model_name)
            if model_name in self.model_dict:
                model, token2id = self.init_model(model_config_lst[model_name])
                self.token2id[model_name] = token2id
                self.model[model_name] = model

    def general_replacement(self, sent):
        sent = data_utils.replace(sent)
        sent = data_utils.full2half(sent)
        sent = data_utils.normal(sent)
        return sent

    def prepare_data(self, question, candidate_lst):
        question_ = self.general_replacement(question)
        question = data_cleaner_api.clean(question)
        question_lst = [cut_tool.cut(question)]*len(candidate_lst)

        candidate_lst = [self.general_replacement(candidate) for candidate in candidate_lst]
        candidate_lst = [cut_tool.cut(data_cleaner_api.clean(candidate)) for candidate in candidate_lst]
        print(candidate_lst)
        return [question_lst, candidate_lst]

    def model_eval(self, model_name, question_lst, candidate_lst):
        
        eval_batch = get_batch_data.get_eval_batches(candidate_lst, 
                                    question_lst, 
                                    100, 
                                    self.token2id[model_name], 
                                    is_training=False)

        eval_probs = []
        sent_repres = []
        eval_labels = []
        for batch in eval_batch:
            print(batch)
            [logits, preds, repres] = self.model[model_name].infer(batch, mode="infer", 
                                                    is_training=False, learning_rate=0.001)
            eval_probs.extend(preds.tolist())
            eval_labels.extend(np.argmax(preds, axis=-1).tolist())
            sent_repres.extend(repres.tolist())
        return eval_probs, eval_labels, sent_repres

    def infer(self, question, candidate_lst):
        print(question, candidate_lst)
        eval_probs, eval_labels, sent_repres = {}, {}, {}
        [question_lst, 
            candidate_lst] = self.prepare_data(question, candidate_lst)
        for model_name in self.model:
            [probs, labels, repres] = self.model_eval(model_name, question_lst, candidate_lst)
            eval_probs[model_name] = probs
            sent_repres[model_name] = repres
            eval_labels[model_name] = labels
        return eval_probs, eval_labels, sent_repres

if __name__ == "__main__":

    from flask import Flask, render_template,request,json
    from flask import jsonify
    import json
    import flask
    from collections import OrderedDict
    import requests
    from pprint import pprint

    app = Flask(__name__)
    timeout = 500

    config = {}
    config["model_config"] = "/notebooks/source/classifynet/model_config.json"
    # config["model_config_path"] = "/data/xuht/test/classify_question_type_focal_loss/esim/logs"
    # config["vocab_path"] = "/data/xuht/question_type/emb_mat.pkl"
    
    # model_config_lst = {}
    # model_config_lst["esim"] = {
    #     "model_name":"esim",
    #     "model_str":"esim_1535683401_7.937636227323642_0.8416456434225326",
    #     "model_dir":"/data/xuht/test/classify_question_type_focal_loss/esim/models"
    # }

    # config["model_config_path"] = "/data/xuht/eventy_detection/center_loss/event_detection/esim/logs"
    # config["model_config_path"] = "/data/xuht/test/classify_tianfeng_speech_command_20180913_focal_loss_big/esim/logs"
    # config["vocab_path"] = "/data/xuht/eventy_detection/center_loss/emb_mat.pkl"
    model_config_lst = {}
    # model_config_lst["esim"] = {
    #     "model_name":"esim",
    #     # "model_str":"esim_1537185831_0.40354261073943754_0.8562106117751447",
    #     # "model_dir":"/data/xuht/test/classify_tianfeng_speech_command_20180913_focal_loss_big/esim/models"
    #     "model_str":"esim_1538096431_0.15536882398801022_1.0",
    #     "model_dir":"/data/xuht/eventy_detection/center_loss/event_detection/esim/models"
    # }
    model_config_lst["esim"] = {
        "model_name":"esim",
        "model_str":"esim_multi_head_no_cudnn_1539960743_0.28559159703611253_0.7836790666693864",
        "model_dir":"/data/xuht/eventy_detection/sentiment/model/esim_multi_head_no_cudnn/models",
        "model_config_path":"/data/xuht/eventy_detection/sentiment/model/esim_multi_head_no_cudnn/logs",
        "vocab_path":"/data/xuht/eventy_detection/sentiment/model/emb_mat.pkl"
    }

    model_config_lst["re_augument"] = {
        "model_name":"re_augument",
        "model_str":"re_augument_with_norule_1540057943_0.2514616994575806_0.7848746886718195",
        "model_dir":"/data/xuht/eventy_detection/sentiment/model/re_augument_with_norule/models",
        "model_config_path":"/data/xuht/eventy_detection/sentiment/model/re_augument_with_norule/logs",
        "vocab_path":"/data/xuht/eventy_detection/sentiment/model/emb_mat.pkl",
        "label_emb":"/data/xuht/eventy_detection/sentiment/model/label_emb.pkl"
    }

    eval_api = Eval(config)
    eval_api.init(model_config_lst)
    def infer(data):
        question = data.get("question", u"为什么头发掉得很厉害")
        candidate_lst = data.get("candidate", ['我头发为什么掉得厉害','你的头发为啥掉这么厉害', 
                    'vpn无法开通', '我的vpn无法开通', '卤面的做法,西红柿茄子素卤面怎么做好吃',
                     '茄子面条卤怎么做'])
        preds, labels, sent_repres = eval_api.infer(question, candidate_lst)
        # for key in preds:
        #     for index, item in enumerate(preds[key]):
        #         preds[key][index] = str(preds[key][index])
        # for key in sent_repres:
        #     for index, item in enumerate(sent_repres[key]):
        #         sent_repres[key][index] = str(sent_repres[key][index].tolist())
        # for key in labels:
        #     for index, item in enumerate(labels[key]):
        #         labels[key][index] = str(labels[key][index].tolist())
        return preds, labels, sent_repres

    @app.route('/classifynet', methods=['POST'])
    def classifynet():
        data = request.get_json(force=True)
        print("=====data=====", data)
        return jsonify(infer(data))

    app.run(debug=False, host="0.0.0.0", port=8011)

    # preds, sent_repres = eval_api.infer([    
    #             "广州客运站的数目",
    #             "广州有多少个客运站",
    #             "广州有几个汽车客运站",
    #             "广州天河有几个客运站",
    #             "广州天河区有几个汽车客运站",
    #             "深圳有几个客运站"])


    # import pickle as pkl
    # pkl.dump(sent_repres, 
    #         open("/data/xuht/test/classifynet/esim/test.pkl", "wb"))