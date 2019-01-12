# -*- coding: UTF-8 -*-
import re
from collections import OrderedDict
import jieba
import codecs
from hanziconv import HanziConv
import os
import string

import json
import jieba.posseg as pseg
import numpy as np

FH_NUM = (
    (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
    (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_NUM = dict(FH_NUM)
FH_ALPHA = (
    (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
    (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
    (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
    (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
    (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"), (u"ｚ", u"z"),
    (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
    (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
    (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
    (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
    (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"), (u"Ｚ", u"Z"),
)
FH_ALPHA = dict(FH_ALPHA)

NUM = (
    (u"一", "1"), (u"二" ,"2"), (u"三", "3"), (u"四", "4"), (u"五", "5"), (u"六", "6"), (u"七", "7"),
    (u"八", "8"), (u"九", "9"), (u"零", "0"), (u"十", "10")
)
NUM = dict(NUM)

CH_PUNCTUATION = u"[＂＃＄％＆＇，：；＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]"
EN_PUNCTUATION = u"['!#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~']"

sub_dicit = {u"老师好":"",
         u"老师":u"", u"你好":u"", u"您好":u"", 
         u"请问":u"", u"请":u"", u"谢谢":u"", 
         u"&quot":u""}

class DataCleaner(object):
    def __init__(self, params_path):
        self.params_path = params_path
        self.read_word()
        self.read_synonym_word()
        self.read_non_words()

    def read_non_words(self):
        word_path = self.params_path.get("non_words", "")
        print("----non word path----", word_path)
        if os.path.exists(word_path):
            with codecs.open(word_path, "r", "utf-8") as f:
                self.non_word = f.read().splitlines()
        else:
            self.non_word = None
        print(self.non_word,"----non word----")

    def calculate_non_word(self, input_string):
        non_cnt = 0
        if self.non_word:
            word_cut = list(jieba.cut(input_string))
            for word in self.non_word:
                if word in word_cut:
                    non_cnt += 1
        if np.mod(non_cnt, 2) == 0:
            return 0
        else:
            return 1

    def synthom_replacement(self, input_string):
        cut_word_list = list(jieba.cut(input_string))
        normalized_word_list = cut_word_list
        for index, word in enumerate(cut_word_list):
            if word in self.synonym_dict:
                normalized_word_list[index] = self.synonym_dict[word]
        return "".join(normalized_word_list)

    def remove_stop_word(self, input_string):
        cut_word_list = list(jieba.cut(input_string))
        normalized_word_list = []
        for word in cut_word_list:
            if word in self.stop_word:
                continue
            else:
                normalized_word_list.append(word)
        return "".join(normalized_word_list)

    def remove_symbol(self, input_string):
        cn_text = re.sub(CH_PUNCTUATION, "", input_string)
        en_text = re.sub(EN_PUNCTUATION, "", cn_text)
        return en_text

    def poc_clean(self, input_string):
        tmp = self.upper2lower(input_string)
        tmp = self.tra2sim(tmp)
        tmp = self.full2half(tmp)

        if self.synonym_dict:
            tmp = self.synthom_replacement(tmp)

        if self.stop_word:
            nonstop_text = self.remove_stop_word(tmp)
            if len(nonstop_text) >= 1:
                tmp = nonstop_text

        non_symbol_text = self.remove_symbol(tmp)
        if len(non_symbol_text) >= 1:
            tmp = non_symbol_text

        char_pattern = re.compile(u"[\u4e00-\u9fa5,0-9,a-z,A-Z]+")
        tmp = "".join(char_pattern.findall(tmp))
        output = ""
        for token in tmp:
            if len(token) >= 1:
                output += token
        return output

    def clean(self, input_string):
        tmp = self.upper2lower(input_string)
        tmp = self.tra2sim(tmp)
        # tmp = self.full2half(tmp)

        return tmp
    
    def read_word(self):
        word_path = self.params_path.get("stop_word", "")
        if os.path.exists(word_path):
            with codecs.open(word_path, "r", "utf-8") as f:
                self.stop_word = f.read().splitlines()
                
        else:
            print("not exiting params_path".format(word_path))
            self.stop_word = None
            
    def read_synonym_word(self):
        self.synonym_dict = {}
        synonym_path = self.params_path.get("synthom_path", "")
        if os.path.exists(synonym_path):
            with codecs.open(synonym_path, "r", "utf-8") as f:
                data = f.read().splitlines()
                for item in data:
                    content = item.split()
                    self.synonym_dict[content[0]] = content[1]
                    print(content[0], content[1])
        else:
            self.synonym_dict = None
        
    def synonym_word_mapping(self):
        self.synonym2standard = OrderedDict()
        for key in self.synonym_dict:
            for item in self.synonym_dict[key]:
                self.synonym2standard[item] = key
    
    def upper2lower(self, input_string):
        return input_string.lower()
    
    def subtoken(self, input_string):
        tmp_string = input_string
        for key in sub_dicit:
            tmp_string = re.sub(key, sub_dicit[key], tmp_string)
        return tmp_string
    
    def lower2upper(self, input_string):
        return input_string.upper()
    
    def replace_phrase(input_string, phrase_dict):
        s = input_string
        for key in phrase_dict.keys():
            s = re.sub(key, phrase_dict[key], s)
        return s
    
    def tra2sim(self, input_string):
        s = HanziConv.toSimplified(input_string)
        return s
    
    def full2half(self, input_string):
        s = ""
        for uchar in input_string:
            if uchar in FH_NUM:
                half_char = FH_NUM[uchar]
            if uchar in FH_ALPHA:
                half_char = FH_ALPHA[uchar]
            else:
                half_char = uchar
            # if uchar in NUM:
            #     half_char = NUM[uchar]
            
            s += half_char
        return s

    def detect_en(self, input_string, 
                  en_pattern=re.compile(u'[\u4e00-\u9fa5]'), 
                  alphabet_pattern=re.compile(u"[a-cA-C]")):
        s = []
        for var in en_pattern.split(input_string.decode("utf-8")):
            if len(var) > 1:
                """
                if len(var) >= 1 it is a word or sentence
                """
                s.append(var)
            elif len(var) == 1:
                """
                if len(var) == 1 it may be a alphabet and usually it is a choice for a given question
                """
                tmp_var = alphabet_pattern.findall(var)
                if len(tmp_var) == 1:
                    s.append(self.upper2lower(var)) 
        return s
    
    def detect_ch(self, input_string, ch_pattern = re.compile(u"[\u4e00-\u9fa5]+")):
        s = ch_pattern.findall(input_string.decode("utf-8"))
        s = " ".join(s)
        return s
    
    def sentence_segmentation(self, input_string, symbol_pattern=re.compile(CH_PUNCTUATION)):
        """
        based on CH_PUNCTUATION to segment sentence
        """
        return symbol_pattern.split(input_string.decode("utf-8"))