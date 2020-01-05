"""
基于机器学习的情感分类
"""

# coding: UTF-8
import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# https://blog.csdn.net/qq_31813549/article/details/79964973#4-%E8%BF%87%E9%87%87%E6%A0%B7%E4%B8%8E%E4%B8%8B%E9%87%87%E6%A0%B7%E7%9A%84%E7%BB%93%E5%90%88
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
import pickle
import time
from gensim.models import Word2Vec


class MLSentimentCLF():
    def __init__(self,pos_path,neg_path,feature='avg_w2v',ngram_range=(1,1),max_features=1000,sampling=False):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.feature = feature
        self.stopwords_path = 'raw_data/dict/stopwords.txt'
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.using_pretrained_w2v = False  # 是否用别人训练好的词向量

        ## 如果用别人的词向量
        self.pretrained_word2vec = \
                    'pretrained_w2v/sgns.BaiduEncyclopedia.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
        self.processed_w2v_bieren = 'processed_data/avg_w2v_bieren.pkl'                   # 训练集对应的词向量形式

        ## 如果自己训练词向量
        self.trained_w2v_ziji_model = 'processed_data/w2v.model'                          # 自己训练的词向量保存的位置
        self.trained_w2v_ziji_data = 'processed_data/w2v.pkl'                             # 数据集对应的词向量形式

        if self.using_pretrained_w2v == True:
            self.w2v_dim = 300
        elif self.using_pretrained_w2v == False:
            self.w2v_dim = 100

        self.sampling = sampling
        self.pos_sentence_len = []
        self.neg_sentence_len = []
        self.X_train, self.X_dev, self.X_test, \
        self.y_train, self.y_dev, self.y_test = self.TrainTestConstruct()
        self.LogCLF()


    def loadStopWords(self):
        """
        加载停用词词表
        :return:
        """
        stop_words_list = []
        with open(self.stopwords_path,'r',encoding = 'gb18030',errors='ignore') as f:
            for line in f:
                stop_words_list.append(line.rstrip('\n'))
        stop_words_list.append('\n')
        stop_words_list.append(' ')
        return stop_words_list

    def cleanSentence(self,sentence):
        """
        清洗文本，去除停用词
        :param sentence:
        :return:
        """
        keywords = jieba.analyse.extract_tags(sentence)
        sentence = re.sub(r'[^\u4e00-\u9fa5]', "", sentence)
        split_sent = jieba.cut(sentence)
        result_sent = ""
        for word in split_sent:
            if word not in self.loadStopWords() and word in keywords:
                result_sent = result_sent + " " + str(word)
                # result_sent_list.append(word)
        return result_sent.lstrip(' ')

    def loadData(self,path,type):
        """
        读取一个文件夹下所有的txt文件中的内容，并保存到一个list中
        :param path: 文件夹所在的路径，如'raw_data/pos/'
        :param type: 数据集类型，正例还是负例
        :return: 文件内容列表与文件数目
        """
        files = os.listdir(path) # ref: https://ask.csdn.net/questions/684269
        result_list = []
        file_num = 0
        for file in files:
            base_file_name = os.path.basename(file)
            full_name = path + base_file_name
            with open (full_name, 'r',encoding = 'gb18030',errors='ignore') as f:
                text = f.read()
                cleaned_text = self.cleanSentence(text)
                result_list.append(cleaned_text)
                if type == 'pos':
                    self.pos_sentence_len.append(len(cleaned_text))
                if type == 'neg':
                    self.neg_sentence_len.append(len(cleaned_text))
            file_num += 1
            # if file_num == 10:
            #     break
        return result_list,file_num

    def word2vec(self,word):
        """
        将一个词转化成对应的词向量
        :param word:
        :return:
        """
        if self.using_pretrained_w2v == True:
            w2v_dict = self.loadW2V()
        elif self.using_pretrained_w2v == False:
            w2v_dict = Word2Vec.load(self.trained_w2v_ziji_model)

        if word not in w2v_dict:
            return np.array([-1]*self.w2v_dim)
        else:
            return w2v_dict[word]

    def avg_w2v(self,all_data):
        """
        将数据集中所有的句子转化成向量，并平均化
        :param all_data:
        :return:
        """
        result = []
        tmp_w2v = [0]*self.w2v_dim               # 300维度
        for sent in all_data:
            words_list = sent.split(' ')
            words_list_num = len(words_list)
            for word in words_list:
                tmp_word_vec = self.word2vec(word)
                if not (tmp_word_vec == [-1]*self.w2v_dim).all():
                    tmp_w2v += tmp_word_vec
            final_w2v = np.array(tmp_w2v)/words_list_num
            result.append(final_w2v)

        if self.using_pretrained_w2v == True:
            pickle.dump(result,open(self.processed_w2v_bieren,"wb"))
        else:
            pickle.dump(result,open(self.trained_w2v_ziji_data,"wb"))

        return result


    def TrainTestConstruct(self):
        """
        构造训练集、验证集、测试集
        :return:
        """
        pos_text_list, pos_text_num = self.loadData(self.pos_path,'pos')
        neg_text_list, neg_text_num = self.loadData(self.neg_path,'neg')
        pos_label = pos_text_num * [1]
        neg_label = neg_text_num * [0]                                  # 标签设置为0、1好呢还是-1、1好呢
        all_data = []
        all_data.extend(pos_text_list)
        all_data.extend(neg_text_list)
        all_label = []
        all_label.extend(pos_label)
        all_label.extend(neg_label)

        all_data = self.transform_data(all_data)

        # print (all_data[0:10])
        print ("所有数据数：",len(all_data))
        X_train,X_test, y_train, y_test = train_test_split(all_data,all_label,test_size=0.3, random_state=233)
        X_train,X_dev,y_train,y_dev = train_test_split(X_train,y_train,test_size=0.3,random_state=233)
        print ("训练集数目：",len(X_train))
        print ("验证集数目：",len(X_dev))
        print ("测试集数目：",len(X_test))
        return X_train,X_dev,X_test,y_train,y_dev,y_test

    def transform_data(self,all_data):
        """
        对数据进行处理，将文本数据转化成向量
        :param all_data:
        :return:
        """
        if self.feature == 'tfidf':
            all_data = self.tfidf(all_data)                                             # 使用tfidf 特征
        elif self.feature == 'avg_w2v':
            if self.using_pretrained_w2v == True:
                if os.path.exists(self.processed_w2v_bieren):
                    all_data = pickle.load(open(self.processed_w2v_bieren,"rb"))              # 保存的预训练好的与训练集对应的
                else:
                    all_data = self.avg_w2v(all_data)
            else:
                if os.path.exists(self.trained_w2v_ziji_data):
                    all_data = pickle.load(open(self.trained_w2v_ziji_data,'rb'))              # 保存好的词向量
                else:
                    self.trainW2V(all_data)
                    all_data = self.avg_w2v(all_data)
        return all_data

    def tfidf(self,all_data):
        """
        将所有的文档按照tfidf模型进行编码
        :param all_data:
        :return:
        """
        ##  tfidf模型
        # https://blog.csdn.net/blmoistawinde/article/details/80816179
        tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                                      ngram_range=self.ngram_range,max_features=self.max_features).fit(all_data)
        all_data = tfidf_model.transform(all_data).todense()
        print("词汇表长度：", len(tfidf_model.vocabulary_))
        return all_data

    # def LenCounter(self,list):
    #     """
    #     对一个list进行简单统计
    #     :param list:
    #     :return:
    #     """
    #     print ("文档长度统计：")
    #     # list_pd = pd.Series(list)
    #     # counter = Counter(list)
    #     # counter = list_pd.value_counts()
    #     # counter.plot('bar')
    #     # plt.show()
    #     print ("文档数：",len(list))
    #     mean = np.mean(list)
    #     print ("长度均值：",mean)
    #     max = np.max(list)
    #     print ("长度最大值：",max)
    #     min = np.min(list)
    #     print ("长度最小值：",min)
    #     return mean,max,min

    def trainW2V(self,all_data):
        """
        用自己的语料库训练一个词向量模型
        :param all_data:
        :return:
        """
        print ("训练词向量中：")
        tmp_all_data = ' '.join(all_data)
        model = Word2Vec(tmp_all_data)
        print ("词向量已训练完成")
        print ("保存词向量：")
        model.save(self.trained_w2v_ziji_model)
        print ("词向量模型已保存")
        return model

    def loadW2V(self):
        """
        将预训练好的词向量记载到一个list在
        :param tmp_pretrained_word2vec:
        :return:
        """
        if self.using_pretrained_w2v == True:
            # don't use f.read,it will incur "MemoryError", ref: https://blog.csdn.net/u014159143/article/details/80360306
            # it is so slow to read the file with "f.readline()"
            # it is still so slow now!
            word2vec_path = self.pretrained_word2vec
            with open(word2vec_path, 'r', encoding='utf-8') as f:
                word2vec_list = f.readlines()
                # word2vec_list.append(f.read())
            word2vec_list = [i.replace("\n", "").rstrip().split(" ") for i in word2vec_list]

            word2vec_dict = {}
            word2vec_list_len = len(word2vec_list)
            for i in range(1, word2vec_list_len):
                word2vec_dict[word2vec_list[i][0]] = np.array([float(i) for i in word2vec_list[i][1:]])

            # print ("the number of words in the dict: ",len(word2vec_dict))

            # Notice: it is "string" in the list!!!!!
            # for i in range(len(word2vec_np[300])):
            #     print (i,":   ",word2vec_np[300][i])
        else:
            word2vec_dict = self.trained_w2v() ###################################################################################
        return word2vec_dict

    def LogCLF(self):
        """
        用逻辑回归模型进行训练
        :return:
        """
        start_time = time.time()
        logistic_model = LogisticRegression(solver='lbfgs')
        X_train = self.X_train
        y_train = self.y_train

        if self.sampling == True:
            ros = RandomOverSampler(random_state=233)
            # sm = SMOTE()
            X_train, y_train = ros.fit_sample(X_train, y_train)

        logistic_model.fit(X_train, y_train)
        train_score = logistic_model.score(self.X_train, self.y_train)
        dev_score = logistic_model.score(self.X_dev, self.y_dev)
        test_score = logistic_model.score(self.X_test,self.y_test)
        print ("训练集准确率：",train_score)
        print ("验证集准确率：",dev_score)
        print ("测试集准确率：",test_score)
        end_time = time.time()
        print("所用时间：", end_time - start_time)
        return logistic_model

if __name__ == "__main__":
    pos_path = 'raw_data/pos/'
    neg_path = 'raw_data/neg/'
    ngram_range = (1, 1)
    max_features = 10000
    print ("n_gram 特征：",ngram_range)
    MLSentimentCLF(pos_path,neg_path,feature='tfidf',max_features=max_features,sampling=True)
