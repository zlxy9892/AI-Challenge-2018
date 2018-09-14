#encoding:utf-8

import jieba
import codecs
import re

class Seg(object):
    
    def __init__(self, file_stopwords="./data/stopword.txt", file_userdict="./data/dict_for_cutword.txt"):
        self.stopwords = []
        self.stopword_filepath = file_stopwords
        self.userdict_filepath = file_userdict
        self.read_in_stopword()
        self.read_in_userdict()

    def read_in_stopword(self): # 读入停用词
        if self.stopword_filepath is not None:
            file_obj = codecs.open(self.stopword_filepath, 'r', 'utf-8')
            while True:
                line = file_obj.readline()
                line = line.strip('\r\n')
                if not line:
                    break
                self.stopwords.append(line)
            file_obj.close()

    def read_in_userdict(self):    # 读入自定义词典
        if self.userdict_filepath is not None:
            jieba.load_userdict(self.userdict_filepath)
    
    def replace_special_words(self, content):
        pass
        return content

    def cut(self, sentence, stopword=True):
        sentence = self.replace_special_words(sentence)
        seg_list = jieba.cut(sentence)  # 切词
        results = []
        if stopword == False:
            results = list(seg_list)
        else:
            for seg in seg_list:
                if seg in self.stopwords and stopword:
                    continue    # 去除停用词
                results.append(seg)
        return results

    def cut_for_search(self, sentence, stopword=True):
        sentence = self.replace_special_words(sentence)
        seg_list = jieba.cut_for_search(sentence)
        results = []
        if stopword == False:
            results = list(seg_list)
        else:
            for seg in seg_list:
                if seg in self.stopwords and stopword:
                    continue    # 去除停用词
                results.append(seg)
        return results
