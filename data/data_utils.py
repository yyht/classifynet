import numpy as np
import pickle as pkl
import codecs, json, os, sys, jieba, re
from jieba import Tokenizer
from jieba.posseg import POSTokenizer
from collections import OrderedDict, Counter

class jieba_api(object):
    def __init__(self):
        print("----------using jieba cut tool---------")

    def init_config(self, config):
        self.config = config
        self.dt = POSTokenizer()

    def build_tool(self):
        dict_path = self.config.get("user_dict", None)
        if dict_path is not None:
            import codecs
            with codecs.open(dict_path, "r", "utf-8") as frobj:
                lines = frobj.read().splitlines()
                for line in lines:
                    content = line.split("\t")
                    self.dt.add_word(content[0], int(content[1]), 
                                content[2])

    def cut(self, text):
        words = list(self.dt.cut(text))
        # print(words, " ".join([word for word in words if len(word) >= 1]))
        return " ".join([word for word in words if len(word) >= 1])

class cut_tool_api(object):
    def __init__(self):
        print("----------using naive cut tool---------")

    def init_config(self, config):
        self.config = config
        self.dt = POSTokenizer()

    def build_tool(self):
        import codecs
        self.word_type = []
        self.cut_flag = True
        try:
            dict_path = self.config.get("user_dict", None)
            with codecs.open(dict_path, "r", "utf-8") as frobj:
                lines = frobj.read().splitlines()
                for line in lines:
                    content = line.split("\t")
                    try:
                        self.dt.add_word(content[0], int(content[1]), 
                                content[2])
                        self.word_type.append(content[2])
                    except:
                        continue
            print("====succeeded in loading dictionary====", dict_path)
            self.word_type = list(set(self.word_type))
            self.word_type = [item for item in self.word_type if len(item) >= 1]
        except:
            self.cut_flag = False
            
    def cut(self, text, target=None):
        out = []
        char_pattern = re.compile(u"[\u4e00-\u9fa5]+")
        word_list = list(self.dt.lcut("".join(text.split())))
        for word in word_list:
            word = list(word)
            if len(word[0]) == 0:
                continue
            if self.cut_flag:
                if word[1] in self.word_type:
                    if target:
                        if word[0] == target:
                            out.append("<target>")
                        else:
                            out.append(word[1])
                    else:
                        out.append(word[1])
                else:
                    char_cn = char_pattern.findall(word[0])
                    if len(char_cn) >= 1:
                        for item in word[0]:
                            if len(item) >= 1:
                                out.append(item)
                    else:
                        if len(word[0]) >= 1:
                            out.append(word[0])
            else:
                char_cn = char_pattern.findall(word[0])
                if len(char_cn) >= 1:
                    for item in word[0]:
                        if len(item) >= 1:
                            out.append(item)
                else:
                    if len(word[0]) >= 1:
                        out.append(word[0])
        return " ".join(out)

def make_dic(sent_list):
    dic = OrderedDict()
    for item in sent_list:
        token_lst = item.split()
        for token in token_lst:
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1
    return dic

def get_idf_embedding(sent_list, token2id):
    sent_num = len(sent_list)
    df_dic = {}
    for index, sent in enumerate(sent_list):
        token_lst = sent.split()
        for token in token_lst:
            if token in df_dic:
                df_dic[token] += 1
            else:
                df_dic[token] = 1

    for token in df_dic:
        df_dic[token] = np.log(sent_num / (df_dic[token]+1e-10)+1e-10)

    idf_emb_mat = np.random.uniform(low=0.0, high=1.0, 
                    size=(len(token2id),1)).astype(np.float32)
    token2id_lst = [(token, token2id[token]) for token in token2id]
    token2id_lst = sorted(token2id_lst, key=lambda item:item[1])
    for item in token2id_lst:
        token = item[0]
        token_id = item[1]
        if token in df_dic:
            idf_emb_mat[token_id][0] = df_dic[token]

    return idf_emb_mat

def read_pretrained_embedding(embedding_path, 
        dic, vocab_path, min_freq=3,
        emb_idf=False, sent_lst=[]):
    if sys.version_info < (3, ):
        w2v = pkl.load(open(embedding_path, "rb"))
    else:
        w2v = pkl.load(open(embedding_path, "rb"), encoding="iso-8859-1")

    word2id, id2word = OrderedDict(), OrderedDict()
    pad_unk = ["<PAD>", "<UNK>", "<S>", "</S>"]
    for index, token in enumerate(pad_unk):
        word2id[token] = index
        id2word[index] = token

    unk_token = []
    pretrained_token = []
    for token in dic:
        if token in w2v:
            if dic[token] >= min_freq:
                pretrained_token.append(token)
        else:
            if dic[token] >= min_freq:
                unk_token.append(token)

    word_id = 4
    for index, token in enumerate(unk_token):
        word2id[token] = word_id
        id2word[word_id] = token
        word_id += 1

    for index, token in enumerate(pretrained_token):
        word2id[token] = word_id
        id2word[word_id] = token
        word_id += 1

    embed_dim = w2v[list(w2v.keys())[0]].shape[0]
    word_mat = np.random.uniform(low=-0.01, high=0.01, 
                                size=(len(word2id), embed_dim)).astype(np.float32)
    for word_id in range(len(word2id)):
        token = id2word[word_id]
        if token in w2v:
            word_mat[word_id] = w2v[token]

    if emb_idf and sent_lst:
        idf_emb_mat = get_idf_embedding(sent_lst, word2id)

        pkl.dump({"token2id":word2id, "id2token":id2word, 
                "embedding_matrix":word_mat,
                "extra_symbol":pad_unk+unk_token,
                "idf_matrix":idf_emb_mat}, 
                open(vocab_path, "wb"), protocol=2)
    else:
        pkl.dump({"token2id":word2id, "id2token":id2word, 
                "embedding_matrix":word_mat,
                "extra_symbol":pad_unk+unk_token}, 
                open(vocab_path, "wb"), protocol=2)


def random_initialize_embedding(dic, vocab_path, min_freq=3, embed_dim=300):
    word2id, id2word = OrderedDict(), OrderedDict()
    pad_unk = ["<PAD>", "<UNK>", "<S>", "</S>"]
    for index, token in enumerate(pad_unk):
        word2id[token] = index
        id2word[index] = token

    word_id = 4
    for index, token in enumerate(dic):
        if dic[token] >= min_freq:
            word2id[token] = word_id
            id2word[word_id] = token
            word_id += 1
    word_mat = np.random.uniform(low=-0.01, high=0.01, 
                                size=(len(word2id), embed_dim)).astype(np.float32)
    pkl.dump({"token2id":word2id, "id2token":id2word, 
            "embedding_matrix":word_mat,
            "extra_symbol":pad_unk}, open(vocab_path, "wb"), protocol=2)

def utt2id(utt, token2id, pad_token, start_token=None, end_token=None):
    utt2id_list = []
    if start_token:
        utt2id_list = [token2id[start_token]]
    for index, word in enumerate(utt.split()):
        utt2id_list.append(token2id.get(word, token2id["<UNK>"]))
    if end_token:
        utt2id_list.append(token2id[end_token])
    return utt2id_list

def read_classify_data(data_path, mode, word_cut_api, 
                data_cleaner_api, split_type="blank"):
    with codecs.open(data_path, "r", "utf-8") as frobj:
        lines = frobj.read().splitlines()
        corpus = []
        gold_label = []
        corpus_len = []
        s = 1
        for line in lines:
            if split_type == "blank":
                content = line.split()
            elif split_type == "tab":
                content = line.split("\t")
            if mode == "train" or mode == "test":
                if len(content) >= 2:
                    try:
                        sent = content[0]
                        label = int(content[1])
                        if s == 1:
                            print(sent, word_cut_api.cut(sent))
                            s += 1
                        sent = data_cleaner_api.clean(sent)
                        corpus.append(word_cut_api.cut(sent))
                        gold_label.append(label)
                        corpus_len.append(len(sent))
                    except:
                        continue
            else:
                if len(content) >= 1:
                    sent = content[0] 
                    sent = data_cleaner_api.clean(sent)
                    corpus.append(word_cut_api.cut(sent))
                    corpus_len.append(len(sent))
        return [corpus, gold_label, corpus_len]

def read_data(data_path, mode, word_cut_api, data_cleaner_api, split_type="blank"):
    tmp = open("/data/xuht/abc.txt", "w")
    with codecs.open(data_path, "r", "utf-8") as frobj:
        lines = frobj.read().splitlines()
        corpus_anchor = []
        corpus_check = []
        gold_label = []
        anchor_len = []
        check_len = []
        s = 1
        for line in lines:
            if split_type == "blank":
                content = line.split()
            elif split_type == "tab":
                content = line.split("\t")
            if mode == "train" or mode == "test":
                if len(content) >= 3:
                    try:
                        sent1 = content[0]
                        sent2 = content[1] 
                        label = int(content[2])

                        cut_sent1 = word_cut_api.cut(sent1)
                        cut_sent2 = word_cut_api.cut(sent2)
                        tmp.write(cut_sent1+"\n")
                        corpus_anchor.append(cut_sent1)
                        corpus_check.append(cut_sent2)
                        gold_label.append(label)
                        if len(cut_sent1.split()) >= 500 or len(cut_sent2.split()) >= 500:
                            continue
                        anchor_len.append(len(cut_sent1.split()))
                        check_len.append(len(cut_sent2.split()))

                        if s == 1:
                            print(cut_sent1, "===",cut_sent2, "===",word_cut_api.cut(sent1))
                            s += 1
                        
                    except:
                        continue
            else:
                if len(content) >= 2:
                    sent1 = content[0]
                    sent2 = content[1] 
                    sent1 = data_cleaner_api.clean(sent1)
                    sent2 = data_cleaner_api.clean(sent2)
                    corpus_anchor.append(word_cut_api.cut(sent1))
                    corpus_check.append(word_cut_api.cut(sent2))
                    anchor_len.append(len(sent1))
                    check_len.append(len(sent2))
        tmp.close()
        return [corpus_anchor, corpus_check, gold_label, anchor_len, check_len]

def utt2charid(utt, token2id, max_length, char_limit):
    utt2char_list = np.zeros([max_length, char_limit])
    for i, word in enumerate(utt.split()):
        for j, char in enumerate(word):
            if char in token2id:
                utt2char_list[i,j] = token2id[char]
            else:
                utt2char_list[i,j] = token2id["<UNK>"]

    return utt2char_list

def id2utt(uttid_list, id2token):
    utt = u""
    for index, idx in enumerate(uttid_list):
        if idx == 0:
            break
        else:
            utt += id2token[idx]
    return utt

import html, re
def replace(doc):
    doc = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)|\\〔.*?〕", "", doc)
    doc = re.sub(r'\d?.?\d+.?\d*(%|百分点)', '<percent>', doc)
    doc = re.sub(r'(（|\()\d+.?\w*(）|\))', '<stock_no>', doc)
    doc = re.sub(r'\d+.?\d*(\w元|百万|千万|亿|万)', '<money>', doc)
    doc = re.sub(r'\d+.?\d+(\d|,)*?(股|\w股)', '<stock_num>', doc)
    doc = re.sub(r'\d+.?\d*\w?(千瓦\w?|\w吨)', '<energe_num>', doc)
    doc = re.sub(r'(\d+年)?\d+月\d+(日|号)', '<date>', doc)
    doc = re.sub(r'\d+年\d+月', '<date>', doc)
    doc = re.sub(r'\d+年-\d+年', '<date>', doc)
    doc = re.sub(r'\d+年', '<date>', doc)
    doc = re.sub(u"\\《.*?》", "<title>", doc)
    doc = re.sub(u"\d+、", "", doc)

    return doc

def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)

def normal(doc):
    doc = doc.lower()
    doc = full2half(doc)

    regx = re.compile(r'(http://|www){1}[a-zA-Z0-9.?/&=:]*',re.S) 
    doc = regx.sub("", doc)

    doc = html.unescape(doc).replace(' ', '') 
    doc = replace(doc)
    return doc

