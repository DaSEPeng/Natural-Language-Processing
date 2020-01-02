"""
基于预训练的词向量计算词的相似度
"""

import numpy as np
import pandas as pd

class Word2vecSimilarity():
    def __init__(self):
        self.word2vec_path = 'sgns.BaiduEncyclopedia.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
        self.word2vec_dict = self.loadWord2vec(self.word2vec_path)

    def loadWord2vec(self,tmp_word2vec_path):
        """
        将预训练的词向量导入一个字典
        :param tmp_word2vec_path: 词向量的路径
        :return: 字典{词：词向量}
        """
        word2vec_list = []
        word2vec_path = tmp_word2vec_path
        with open(word2vec_path, 'r', encoding='utf-8') as f:
            word2vec_list = f.readlines()
            # word2vec_list.append(f.read())
        word2vec_list = [i.replace("\n", "").rstrip().split(" ") for i in word2vec_list]

        word2vec_dict = {}
        word2vec_list_len = len(word2vec_list)
        for i in range(1, word2vec_list_len):
            word2vec_dict[word2vec_list[i][0]] = np.array([float(i) for i in word2vec_list[i][1:]])
        return word2vec_dict

    def getW2V(self,word):
        return self.word2vec_dict[word]

    def cos_sim(self,v1,v2):
        num = float(np.sum(v1 * v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        result = 0.5 + 0.5 * num/denom
        return result

    def euclidean_sim(self,v1,v2):
        return 1/(1 + np.linalg.norm(v1-v2))

    def w2v_sim(self,word1,word2,type):
        v1 = self.getW2V(word1)
        v2 = self.getW2V(word2)
        if type == 'cos':
            return self.cos_sim(v1,v2)
        elif type == 'eu':
            return self.euclidean_sim(v1,v2)
        else:
            return -1



