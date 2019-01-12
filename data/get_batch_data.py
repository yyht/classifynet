from data.data_utils import utt2id
import numpy as np

def drop_word(sent, word_drop):
    curr_sent = []
    mask = np.random.uniform(low=0.0, high=1.0, size=len(sent)) > word_drop
    for index, keep in enumerate(mask):
        if keep:
            curr_sent.append(sent[index])
    if len(curr_sent) <= 0:
        curr_sent = sent
    return curr_sent

def dynamic_padding(corpus, token2id, pad_token="<PAD>", 
                    start_token=None, end_token=None,
                    if_word_drop=None, word_drop_rate=0.8,
                    max_anchor_len=500, if_max_anchor_len=False,
                    max_check_len=20, if_max_check_len=False,
                    if_anchor=False):
    max_len_corpus = 0
    corpus_lst = []
    for utt in corpus:
        sent_lst = utt2id(utt, token2id, pad_token, start_token, end_token)
        if if_word_drop:
            sent_lst = drop_word(sent_lst, word_drop_rate)
        corpus_lst.append(sent_lst)
        if max_len_corpus < len(sent_lst):
            max_len_corpus = len(sent_lst)
    if if_max_anchor_len and if_anchor == True:
        max_len_finial = max_anchor_len
    elif if_max_check_len and if_anchor == False:
        max_len_finial = max_check_len
    else:
        max_len_finial = max_len_corpus
    for index, sent_lst in enumerate(corpus_lst):
        if len(sent_lst) > max_len_finial:
            corpus_lst[index] = corpus_lst[index][0:max_len_finial]
        else:
            corpus_lst[index] += [token2id[pad_token]]*(max_len_finial-len(sent_lst))
    return corpus_lst

def get_eval_classify_batches(corpus, batch_size, 
                    token2id, is_training=True, 
                    if_word_drop=None, word_drop_rate=0.8,
                    max_anchor_len=500, if_max_anchor_len=False,
                    max_check_len=20, if_max_check_len=False,
                    if_anchor=False):
    if is_training:
        shuffled_index = np.random.permutation(len(corpus))
    else:
        shuffled_index = range(len(corpus))

    batch_num = int(len(corpus) / batch_size)
    end_index = 0
    for index in range(batch_num):
        start_index = index * batch_size
        end_index = start_index + batch_size

        sub_corpus = [corpus[t] for t in shuffled_index[start_index:end_index]]
        corpus_lst_ = dynamic_padding(sub_corpus, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        corpus_lst = []
        for corpus_ in corpus_lst_:
            if sum(corpus_) == 0:
                continue
            else:
                corpus_lst.append(corpus_)

        corpus_lst = np.asarray(corpus_lst).astype(np.int32)

        yield corpus_lst, []

    if end_index < len(corpus):

        sub_corpus = [corpus[t] for t in shuffled_index[end_index:]]
        corpus_lst_ = dynamic_padding(sub_corpus, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        corpus_lst = []
        for corpus_ in corpus_lst_:
            if sum(corpus_) == 0:
                continue
            else:
                corpus_lst.append(corpus_)

        corpus_lst = np.asarray(corpus_lst).astype(np.int32)

        yield corpus_lst, []

def get_eval_batches(anchor, check, batch_size, 
                    token2id, is_training=True,
                    if_word_drop=None, 
                    word_drop_rate=0.8,
                    max_anchor_len=500, if_max_anchor_len=False,
                    max_check_len=20, if_max_check_len=False):
    if is_training:
        shuffled_index = np.random.permutation(len(anchor))
    else:
        shuffled_index = range(len(anchor))
    batch_num = int(len(anchor) / batch_size)
    end_index = 0
    for index in range(batch_num):
        start_index = index * batch_size
        end_index = start_index + batch_size

        sub_anchor = [anchor[t] for t in shuffled_index[start_index:end_index]]
        sub_check = [check[t] for t in shuffled_index[start_index:end_index]]

        anchor_lst = dynamic_padding(sub_anchor, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        check_lst = dynamic_padding(sub_check, token2id,
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=False)

        anchor_lst = np.asarray(anchor_lst).astype(np.int32)
        check_lst = np.asarray(check_lst).astype(np.int32)

        yield anchor_lst, check_lst, []

    if end_index < len(anchor):

        sub_anchor = [anchor[t] for t in shuffled_index[end_index:]]
        sub_check = [check[t] for t in shuffled_index[end_index:]]

        anchor_lst = dynamic_padding(sub_anchor, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        check_lst = dynamic_padding(sub_check, token2id,
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=False)

        anchor_lst = np.asarray(anchor_lst).astype(np.int32)
        check_lst = np.asarray(check_lst).astype(np.int32)

        yield anchor_lst, check_lst, []

def get_batches(anchor, check, label, batch_size, 
                    token2id, is_training=True,
                    if_word_drop=None, word_drop_rate=0.8,
                    max_anchor_len=500, if_max_anchor_len=False,
                    max_check_len=20, if_max_check_len=False):

    if is_training:
        shuffled_index = np.random.permutation(len(anchor))
    else:
        shuffled_index = range(len(anchor))
    batch_num = int(len(anchor) / batch_size)
    end_index = 0
    for index in range(batch_num):
        start_index = index * batch_size
        end_index = start_index + batch_size

        sub_anchor = [anchor[t] for t in shuffled_index[start_index:end_index]]
        sub_check = [check[t] for t in shuffled_index[start_index:end_index]]

        label_lst = [label[t] for t in shuffled_index[start_index:end_index]]
        anchor_lst = dynamic_padding(sub_anchor, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        check_lst = dynamic_padding(sub_check, token2id,
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=False)

        label_lst = np.asarray(label_lst).astype(np.int32)
        anchor_lst = np.asarray(anchor_lst).astype(np.int32)
        check_lst = np.asarray(check_lst).astype(np.int32)

        yield anchor_lst, check_lst, label_lst

    if end_index < len(anchor):

        sub_anchor = [anchor[t] for t in shuffled_index[end_index:]]
        sub_check = [check[t] for t in shuffled_index[end_index:]]

        label_lst = [label[t] for t in shuffled_index[end_index:]]
        anchor_lst = dynamic_padding(sub_anchor, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        check_lst = dynamic_padding(sub_check, token2id,
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=False)

        label_lst = np.asarray(label_lst).astype(np.int32)
        anchor_lst = np.asarray(anchor_lst).astype(np.int32)
        check_lst = np.asarray(check_lst).astype(np.int32)

        yield anchor_lst, check_lst, label_lst

def get_classify_batch(corpus, label, batch_size, 
                    token2id, is_training=True,
                    if_word_drop=None, word_drop_rate=0.8,
                    max_anchor_len=500, if_max_anchor_len=False,
                    max_check_len=20, if_max_check_len=False):
    
    if is_training:
        shuffled_index = np.random.permutation(len(corpus))
    else:
        shuffled_index = range(len(corpus))

    batch_num = int(len(corpus) / batch_size)
    end_index = 0
    for index in range(batch_num):
        start_index = index * batch_size
        end_index = start_index + batch_size

        sub_corpus = [corpus[t] for t in shuffled_index[start_index:end_index]]

        label_lst_ = [label[t] for t in shuffled_index[start_index:end_index]]
        corpus_lst_ = dynamic_padding(sub_corpus, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        corpus_lst, label_lst = [], []
        for corpus_, label_ in zip(corpus_lst_, label_lst_):
            if sum(corpus_) == 0:
                continue
            else:
                corpus_lst.append(corpus_)
                label_lst.append(label_)

        label_lst = np.asarray(label_lst).astype(np.int32)
        corpus_lst = np.asarray(corpus_lst).astype(np.int32)

        yield corpus_lst, label_lst

    if end_index < len(corpus):

        sub_corpus = [corpus[t] for t in shuffled_index[end_index:]]

        label_lst_ = [label[t] for t in shuffled_index[end_index:]]
        corpus_lst_ = dynamic_padding(sub_corpus, token2id, 
                                if_word_drop=if_word_drop, 
                                word_drop_rate=word_drop_rate,
                                max_anchor_len=max_anchor_len, 
                                if_max_anchor_len=if_max_anchor_len,
                                max_check_len=max_check_len, 
                                if_max_check_len=if_max_check_len,
                                if_anchor=True)

        corpus_lst, label_lst = [], []
        for corpus_, label_ in zip(corpus_lst_, label_lst_):
            if sum(corpus_) == 0:
                continue
            else:
                corpus_lst.append(corpus_)
                label_lst.append(label_)

        label_lst = np.asarray(label_lst).astype(np.int32)
        corpus_lst = np.asarray(corpus_lst).astype(np.int32)

        yield corpus_lst, label_lst