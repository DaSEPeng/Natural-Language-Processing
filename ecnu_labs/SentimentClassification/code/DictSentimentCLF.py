"""
基于词典的情感分类
"""

import os
import pandas as pd
import jieba
# jieba.load_userdict("raw_data/dict/all.txt")  需要标注，不能这么做

class DictSentimentCLF():
    def __init__(self):
        self.root_path = 'raw_data/'
        self.stop_words_path = self.root_path + 'stopwords.txt'

    def CLF(self,path):
        """
        读取一个文件夹下所有的txt文件中的内容，并保存到一个list中
        :param path: 文件夹所在的路径，如'raw_data/pos/'
        :return: 文件内容列表与文件数目
        """
        files = os.listdir(path) # ref: https://ask.csdn.net/questions/684269
        result_list = []
        file_num = 0
        for file in files:
            base_file_name = os.path.basename(file)
            full_name = data_path + base_file_name
            with open (full_name, 'r',encoding = 'gb18030',errors='ignore') as f:
                # 注意此处的编码方式，否则会报 illegal multibyte sequence 的错误
                # 参考：https://blog.csdn.net/jiasudu1234/article/details/71173281/
                text = f.read()
                senti = self.SentenceSentiment(text)
                # print ("第", file_num , "个文档","情感：",senti)
                result_list.append(senti)
            file_num += 1
            # if file_num == 10:
            #     break
        print ("The Number of The Processed Files：",file_num)
        return result_list,file_num

    def loadStopWords(self):
        """
        加载停用词
        :return: 停用词list
        """
        stop_words_list = []
        with open(self.stop_words_path,'r',encoding = 'gb18030',errors='ignore') as f:
            for line in f:
                stop_words_list.append(line.rstrip('\n'))
        stop_words_list.append('\n')
        return stop_words_list

    def SentenceSentiment(self,sentence):
        """
        对单个句子进行情感分类
        :param sentence:
        :return:
        """
        split_sent = jieba.cut(sentence)
        tmp_degree = 1
        tmp_senti = 0
        for word in split_sent:
            # print ("当前词： ", word)
            word_degree = self.calDegree(word)
            word_senti = self.wordSenti(word)
            if word_senti == 0:                         # 中性词或者程度副词
                tmp_degree *= word_degree
            elif word_senti != 0:                       # 情感词
                tmp_senti += tmp_degree * word_senti
                tmp_degree == 1
        #     print ("当前权重值：",tmp_degree)
        # print ("情感加权值：",tmp_senti)
        if tmp_senti == 0:
            return 0
        if tmp_senti > 0:
            return 1
        if tmp_senti < 0:
            return -1

    def loadDict(self,path):
        with open(path, 'r',encoding = 'gb18030',errors='ignore') as f:
            text = f.read()
            word_list = [i.rstrip(" ") for i in text.split('\n')]
        return word_list

    def calDegree(self,word):
        """
        计算一个词的程度
        :param word: 词
        :return: 程度，一个预先定义的数值，否定词定义为-1
        """
        root_path = 'raw_data/dict/'
        chengdu_9_list = self.loadDict(root_path + 'chengdu_9.txt')
        if word in chengdu_9_list:
            return 9
        chengdu_8_list = self.loadDict(root_path + 'chengdu_8.txt')
        if word in chengdu_8_list:
            return 8
        chengdu_7_list = self.loadDict(root_path + 'chengdu_7.txt')
        if word in chengdu_7_list:
            return 7
        chengdu_5_list = self.loadDict(root_path + 'chengdu_5.txt')
        if word in chengdu_5_list:
            return 5
        chengdu_3_list = self.loadDict(root_path + 'chengdu_3.txt')
        if word in chengdu_3_list:
            return 3
        chengdu_2_list = self.loadDict(root_path + 'chengdu_2.txt')
        if word in chengdu_2_list:
            return 2
        if word == '不':
            return -1
        else:
            return 1

    def wordSenti(self,word):
        root_path = 'raw_data/dict/'
        pos_qinggan = self.loadDict(root_path + 'pos_qinggan.txt')
        if word in pos_qinggan:
            return 1
        neg_qinggan = self.loadDict(root_path + 'neg_qinggan.txt')
        if word in neg_qinggan:
            return -1
        else:
            return 0

    def acc(self,pred_list,text_num,senti_label):
        acc_num = 0
        for pred_label in pred_list:
            if pred_label == senti_label:
                acc_num += 1
            else:
                pass
        print ("准确率：",acc_num/text_num)

ld = DictSentimentCLF()

print ("处理正例样本：")
data_path = 'raw_data/pos/'
text_list , text_num = ld.CLF(data_path)
ld.acc(text_list,text_num,senti_label=1)
pd.Series(text_list).to_csv("pos_result.csv")

print ("处理负例样本：")
data_path = 'raw_data/neg/'
text_list , text_num = ld.CLF(data_path)
ld.acc(text_list,text_num,senti_label=-1)
pd.DataFrame(text_list).to_csv("neg_result.csv")




