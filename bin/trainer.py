import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse
from bunch import Bunch

from random import random

import sys,os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append("..")

from model.esim.esim import ESIM
from model.biblosa.biblosa import BiBLOSA
from model.transformer.base_transformer import BaseTransformer
from model.transformer.universal_transformer import UniversalTransformer
from model.capsule.capsule import Capsule
from model.label_network.label_network import LabelNetwork
from model.leam.leam import LEAM
from model.swem.swem import SWEM
from model.textcnn.textcnn import TextCNN
from model.deeppyramid.deeppyramid import DeepPyramid
from model.re_augument.re_augument import ReAugument

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_clearner_api = data_clean.DataCleaner({})


def prepare_data(data_path, w2v_path, vocab_path, make_vocab=True,
                elmo_w2v_path=None,
                elmo_pca=False, 
                emb_idf=False):

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
        if not elmo_w2v_path:
            data_utils.read_pretrained_embedding(
                            w2v_path, dic, vocab_path, min_freq=1,
                            emb_idf=emb_idf,
                            sent_lst=anchor+check)
        else:
            data_utils.read_pretrained_elmo_embedding(w2v_path, dic, 
                            vocab_path, min_freq=1,
                            elmo_embedding_path=elmo_w2v_path,
                            elmo_pca=elmo_pca,
                            emb_idf=emb_idf,
                            sent_lst=anchor+check)

    if sys.version_info < (3, ):
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
    else:
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                encoding="iso-8859-1")

    return [anchor, check, label, 
            anchor_len, check_len, 
            embedding_info]

def train(config):
    model_config_path = config["model_config_path"]
    FLAGS = namespace_utils.load_namespace(model_config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu_id", "")
    train_path = config["train_path"]
    w2v_path = config["w2v_path"]
    vocab_path = config["vocab_path"]
    dev_path = config["dev_path"]
    elmo_w2v_path = config.get("elmo_w2v_path", None)
    label_emb_path = config.get("label_emb_path", None)

    if label_emb_path:
        import pickle as pkl
        label_emb_mat = pkl.load(open(label_emb_path, "rb"))

    model_dir = config["model_dir"]
    try:
        model_name = FLAGS["output_folder_name"]
    except:
        model_name = config["model"]
    print(model_name, "====model name====")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.mkdir(os.path.join(model_dir, model_name))

    if not os.path.exists(os.path.join(model_dir, model_name, "logs")):
        os.mkdir(os.path.join(model_dir, model_name, "logs"))

    if not os.path.exists(os.path.join(model_dir, model_name, "models")):
        os.mkdir(os.path.join(model_dir, model_name, "models"))

    json.dump(FLAGS, open(os.path.join(model_dir, model_name, "logs", model_name+".json"), "w"))
    logger = logger_utils.get_logger(os.path.join(model_dir, model_name, "logs","log.info"))

    [train_anchor, 
    train_check, 
    train_label, 
    train_anchor_len, 
    train_check_len, 
    embedding_info] = prepare_data(train_path, 
                        w2v_path, vocab_path,
                        make_vocab=True,
                        elmo_w2v_path=elmo_w2v_path,
                        elmo_pca=FLAGS.elmo_pca,
                        emb_idf=config.emb_idf)

    [dev_anchor, 
    dev_check, 
    dev_label, 
    dev_anchor_len, 
    dev_check_len, 
    embedding_info] = prepare_data(dev_path, 
                        w2v_path, vocab_path,
                        make_vocab=False,
                        elmo_w2v_path=elmo_w2v_path,
                        elmo_pca=FLAGS.elmo_pca,
                        emb_idf=config.emb_idf)

    token2id = embedding_info["token2id"]
    id2token = embedding_info["id2token"]
    embedding_mat = embedding_info["embedding_matrix"]
    extral_symbol = embedding_info["extra_symbol"]

    if config.emb_idf:
        idf_emb_mat = embedding_info["idf_matrix"]
        FLAGS.idf_emb_mat = idf_emb_mat
        FLAGS.with_idf = True

    FLAGS.token_emb_mat = embedding_mat
    FLAGS.char_emb_mat = 0
    FLAGS.vocab_size = embedding_mat.shape[0]
    FLAGS.char_vocab_size = 0
    FLAGS.emb_size = embedding_mat.shape[1]
    FLAGS.extra_symbol = extral_symbol
    FLAGS.class_emb_mat = label_emb_mat

    if FLAGS.scope == "ESIM":
        model = ESIM()
    elif FLAGS.scope == "BiBLOSA":
        model = BiBLOSA()
    elif FLAGS.scope == "BaseTransformer":
        model = BaseTransformer()
    elif FLAGS.scope == "UniversalTransformer":
        model = UniversalTransformer()
    elif FLAGS.scope == "Capsule":
        model = Capsule()
    elif FLAGS.scope == "LabelNetwork":
        model = LabelNetwork()
    elif FLAGS.scope == "LEAM":
        model = LEAM()
    elif FLAGS.scope == "SWEM":
        model = SWEM()
    elif FLAGS.scope == "TextCNN":
        model = TextCNN()
    elif FLAGS.scope == "DeepPyramid":
        model = DeepPyramid()
    elif FLAGS.scope == "ReAugument":
        model = ReAugument()

    if FLAGS.scope in ["Capsule", "DeepPyramid"]:
        max_anchor_len = FLAGS.max_length
        max_check_len = 1
        if_max_anchor_len = True,
        if_max_check_len = True
    else:
        max_anchor_len = FLAGS.max_length
        max_check_len = 1
        if_max_anchor_len = False
        if_max_check_len = False

    model.build_placeholder(FLAGS)
    model.build_op()
    model.init_step()

    best_dev_f1 = 0.0
    best_dev_loss = 100.0
    learning_rate = FLAGS.learning_rate
    toleration = 1000
    toleration_cnt = 0
    print("=======begin to train=========")
    for epoch in range(FLAGS.max_epochs):
        train_loss, train_accuracy = 0, 0
        train_data = get_batch_data.get_batches(train_anchor, 
            train_check, 
            train_label, FLAGS.batch_size, 
            token2id, is_training=True,
            max_anchor_len=max_anchor_len, 
            if_max_anchor_len=if_max_anchor_len,
            max_check_len=max_check_len, 
            if_max_check_len=if_max_check_len)

        nan_data = []
        cnt = 0
        train_accuracy_score, train_precision_score, train_recall_score = 0, 0 ,0
        train_label_lst, train_true_lst = [], []

        for index, corpus in enumerate(train_data):
            anchor, entity, label = corpus
            assert entity.shape[-1] == 1
            try:
                [loss, _, global_step, 
                accuracy, preds] = model.step(
                                    [anchor, entity, label], 
                                    is_training=True,
                                    learning_rate=learning_rate)

                import math
                if math.isnan(loss):
                    print(anchor, entity, label, loss, "===nan loss===")
                    break
                train_label_lst += np.argmax(preds, axis=-1).tolist()
                train_true_lst += label.tolist()

                train_loss += loss*anchor.shape[0]
                train_accuracy += accuracy*anchor.shape[0]
                cnt += anchor.shape[0]
            except:
                continue

        train_loss /= float(cnt)

        train_accuracy = accuracy_score(train_true_lst, train_label_lst)
        train_recall = recall_score(train_true_lst, train_label_lst, average='macro')
        train_precision = precision_score(train_true_lst, train_label_lst, average='macro')
        train_f1 = f1_score(train_true_lst, train_label_lst, average='macro')

        info = OrderedDict()
        info["epoch"] = str(epoch)
        info["train_loss"] = str(train_loss)
        info["train_accuracy"] = str(train_accuracy)
        info["train_f1"] = str(train_f1)

        logger.info("epoch\t{}\ttrain\tloss\t{}\taccuracy\t{}\tf1\t{}".format(epoch, train_loss, train_accuracy, train_f1))

        dev_data = get_batch_data.get_batches(dev_anchor, 
            dev_check, 
            dev_label, FLAGS.batch_size, 
            token2id, 
            is_training=False,
            max_anchor_len=max_anchor_len, 
            if_max_anchor_len=if_max_anchor_len,
            max_check_len=max_check_len, 
            if_max_check_len=if_max_check_len)

        dev_loss, dev_accuracy = 0, 0
        cnt = 0
        dev_label_lst, dev_true_lst = [], []
        for index, corpus in enumerate(dev_data):
            anchor, entity, label = corpus

            try:
                [loss, logits, 
                pred_probs, accuracy] = model.infer(
                                    [anchor, entity, label], 
                                    mode="test",
                                    is_training=False,
                                    learning_rate=learning_rate)

                dev_label_lst += np.argmax(pred_probs, axis=-1).tolist()
                dev_true_lst += label.tolist()

                import math
                if math.isnan(loss):
                    print(anchor, entity, pred_probs, index)

                dev_loss += loss*anchor.shape[0]
                dev_accuracy += accuracy*anchor.shape[0]
                cnt += anchor.shape[0]
            except:
                continue
           
        dev_loss /= float(cnt)

        dev_accuracy = accuracy_score(dev_true_lst, dev_label_lst)
        dev_recall = recall_score(dev_true_lst, dev_label_lst, average='macro')
        dev_precision = precision_score(dev_true_lst, dev_label_lst, average='macro')
        dev_f1 = f1_score(dev_true_lst, dev_label_lst, average='macro')

        info["dev_loss"] = str(dev_loss)
        info["dev_accuracy"] = str(dev_accuracy)
        info["dev_f1"] = str(dev_f1)

        logger.info("epoch\t{}\tdev\tloss\t{}\taccuracy\t{}\tf1\t{}".format(epoch, dev_loss, 
                                                        dev_accuracy, dev_f1))

        if dev_f1 > best_dev_f1 or dev_loss < best_dev_loss:
            timestamp = str(int(time.time()))
            model.save_model(os.path.join(model_dir, model_name, "models"), model_name+"_{}_{}_{}".format(timestamp, dev_loss, dev_f1))
            best_dev_f1 = dev_f1
            best_dev_loss = dev_loss

            toleration_cnt = 0

            info["best_dev_loss"] = str(dev_loss)
            info["dev_f1"] = str(dev_f1)

            logger_utils.json_info(os.path.join(model_dir, model_name, "logs", "info.json"), info)
            logger.info("epoch\t{}\tbest_dev\tloss\t{}\tbest_accuracy\t{}\tbest_f1\t{}".format(epoch, dev_loss, 
                                                                                            dev_accuracy, best_dev_f1))
        else:
            toleration_cnt += 1
            if toleration_cnt == toleration:
                toleration_cnt = 0
                learning_rate *= 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_config', type=str, help='model config path')
    parser.add_argument('--model_dir', type=str, help='model path')
    parser.add_argument('--config_prefix', type=str, help='config path')
    parser.add_argument('--gpu_id', type=str, help='gpu id')
    parser.add_argument('--train_path', type=str, help='train data path')
    parser.add_argument('--dev_path', type=str, help='dev data path')
    parser.add_argument('--w2v_path', type=str, help='pretrained w2v path')
    parser.add_argument('--vocab_path', type=str, help='vocab_path')
    parser.add_argument('--label_emb_path', type=str, help='label_emb_path')
    parser.add_argument('--user_dict_path', type=str, help='user_dict_path')
    parser.add_argument('--emb_idf', type=str, help='emb_idf')

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
    config["train_path"] = args.train_path
    config["w2v_path"] = args.w2v_path
    config["vocab_path"] = args.vocab_path
    config["dev_path"] = args.dev_path
    config["label_emb_path"] = args.label_emb_path
    if args.emb_idf == "1":
        config["emb_idf"] = True
    else:
        config["emb_idf"] = False

    cut_tool = data_utils.cut_tool_api()
    cut_tool.init_config({
            "user_dict":args.user_dict_path})
    cut_tool.build_tool()
    config = Bunch(config)
    train(config)


    










