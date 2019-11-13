"""
Preprocess the raw data and extract the feactures. The following types of feature are extracted:
    - count2vec
    - tfidf2vec
    - avg_word2vec
    - avg_word2vec_based_on_tfidf

author: Peng Lee
date: 2019/10/22
contact: ruhao9805@163.com
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
from jieba import analyse                        # Keywords Extraction Tool
from scipy import sparse                         # https://blog.csdn.net/weixin_36218261/article/details/78297716
                                                  # https://blog.csdn.net/winycg/article/details/80967112
import time

RawDataFolder = 'data/raw_data/'
PreprocessedDataFolder = 'data/preprocessed_data/'

# https://github.com/Embedding/Chinese-Word-Vectors
PretrainedWord2VecFolder = 'data/pretrained_word2vec/'
pretrained_word2vec_path = PretrainedWord2VecFolder + \
                      'sgns.BaiduEncyclopedia.target.word-word.dynwin5.thr10.neg5.dim300.iter5'  # or 'sgns.renmin.word'

StopwordsPath = 'stopwords.txt'

################################################## LOAD DATA ##########################################################

def loadDataBase(tmp_data_path):
    """
    Load the data in a file
    :param tmp_data_path:
    :return:
    """
    folders = os.listdir(tmp_data_path)
    tmp_data_X = []
    tmp_data_y = []
    for folder in folders:
        folder_name = os.path.basename(folder)  # i.e. C7-History
        tmp_category = folder_name.split('-')[1]  # i.e. History
        tmp_subfolder = tmp_data_path + folder_name + '/'
        docs = os.listdir(tmp_subfolder)
        for doc in docs:
            doc_base_name = os.path.basename(doc)  # i.e. C7-History003.txt
            full_name = tmp_subfolder + doc_base_name
            with open (full_name,'r', encoding='GB2312',errors="ignore") as f:  # notice the "ignore" action
                tmp_text = f.read()
                # print (chardet.detect(tmp_text))
                tmp_data_X.append(tmp_text)
                tmp_data_y.append(tmp_category)   # don't use extend, it is sill a string here.
    return tmp_data_X,tmp_data_y

def loadData(tmp_train_data_path, tmp_test_data_path):
    """
    Load the train and test data
    :param tmp_train_data_path:
    :param tmp_test_data_path:
    :return:
    """
    tmp_train_X,tmp_train_y = loadDataBase(tmp_train_data_path)
    tmp_test_X,tmp_test_y = loadDataBase(tmp_test_data_path)

    # notice the following transformation
    category_list =  ['Economy', 'Computer', 'Sports', 'Enviornment', 'Politics',
     'Agriculture', 'Art', 'Space', 'History', 'Military', 'Education',
     'Transport', 'Law', 'Medical', 'Philosophy', 'Mine', 'Literature',
     'Energy', 'Electronics', 'Communication']
    category_mapping = dict(zip(category_list,range(20)))
    print ("Category Mapping:\n",category_mapping)

    tmp_train_y = [category_mapping[i] for i in tmp_train_y]
    tmp_test_y = [category_mapping[i] for i in tmp_test_y]

    if len(tmp_train_X) == len(tmp_train_y):
        print ("Train Data Size: ", len(tmp_train_y))
    if len(tmp_test_X) == len(tmp_test_y):
        print ("Test Data Size: ", len(tmp_test_y))

    return tmp_train_X,tmp_train_y,tmp_test_X,tmp_test_y

################################################### LOAD STOPWORDS ###################################################

def loadStopwordsList(tmp_stopwords_path):
    """
    Load stopwords to a list
    :param tmp_stopwords_path:
    :return:
    """
    tmp_stopwords_list = []
    with open(tmp_stopwords_path, 'r', encoding='utf-8') as f:
        tmp_stopwords_list.extend(f.readlines())
    stopwords_list = [i.replace("\n",'') for i in tmp_stopwords_list]
    return stopwords_list

############################################## DOCUMENTS STATISTIC #################################################


def docsStat(tmp_train_y_list,tmp_test_y_list):
    """
    Exploratory Data Analysis
    :param tmp_train_y_list:
    :param tmp_test_y_list:
    :return:
    """
    # https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    tmp_train_y_pd = pd.Series(tmp_train_y_list)
    tmp_test_y_pd = pd.Series(tmp_test_y_list)

    tmp_train_y_pd_count = tmp_train_y_pd.value_counts().sort_index()
    tmp_test_y_pd_count = tmp_test_y_pd.value_counts().sort_index()
    print ("Train Data Stat: \n", tmp_train_y_pd_count)
    print("Test Data Stat: \n",tmp_test_y_pd_count)

    labels = range(len(tmp_train_y_pd_count))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, np.array(tmp_train_y_pd_count), width, label='Train Data')
    rects2 = ax.bar(x + width/2, np.array(tmp_test_y_pd_count), width, label='Test Data')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number')
    ax.set_xlabel('Category')
    ax.set_title('Number of Articles in Each Category')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.show()
    fig.savefig('stat.eps', dpi=600, format='eps')
    return 0

##################################################### TEXT PREPROCESSING #############################################

def textPreprocess(tmp_docs_list,k_num = 30,tmp_stopwords_path = StopwordsPath):
    """
    Text Preprocession, clean, extract the keywords, transform the docs to a speacial style: “你好 我是 小明”
    :param tmp_docs_list:
    :param k_num:
    :param tmp_stopwords_path:
    :return:
    """
    doc_num = len(tmp_docs_list)
    preprocessed_docs = []
    stopwords_list = loadStopwordsList(tmp_stopwords_path)           # load stopwords

    for i in range(doc_num):
        if (i%1000 == 0):
            print ("Processing No.", i, "....")
        tmp_doc = tmp_docs_list[i]                                   # string
        p = re.compile('[\u4e00-\u9fa5]')                            # extract only Chinese
        res = re.findall(p, tmp_doc)
        tmp_doc_cleaned = ''.join(res)                               # Chinese Only
        cut_list = list(jieba.cut(tmp_doc_cleaned))

        keywords_tfidf = analyse.extract_tags(tmp_doc_cleaned, topK=k_num)    # https://github.com/fxsjy/jieba

        seged_docs_list = []
        for term in cut_list:
            if term not in stopwords_list and term in keywords_tfidf:
                seged_docs_list.append(term)

        seged_docs = " ".join(seged_docs_list)
        preprocessed_docs.append(seged_docs)
    return preprocessed_docs

############################################ FEATURE EXTRACTION ALGORITHM ###########################################

def Count2Vec(tmp_corpus,tmp_ngram_range =(1,1)):
    """
    Count2vec basd doc2vec
    :param tmp_corpus:
    :param tmp_ngram_range:
    :return:
    """
    print ("Count2Vecing ...")
    vec = CountVectorizer(ngram_range=tmp_ngram_range)
    tmp_X = vec.fit_transform(tmp_corpus)
    print("Count2Vec shape: ", tmp_X.toarray().shape)
    return tmp_X.toarray()


def TFIDF2Vec(tmp_corpus,tmp_ngram_range =(1,1)):
    """
    Tfidf based doc2vec
    :param tmp_corpus:
    :param tmp_ngram_range:
    :return:
    """
    print ("\nTFIDF2Vecing ...")
    vec = TfidfVectorizer(ngram_range=tmp_ngram_range)
    tmp_X = vec.fit_transform(tmp_corpus)
    print("Tfidf2Vec shape: ", tmp_X.toarray().shape)
    return tmp_X.toarray()

def loadW2V(tmp_pretrained_word2vec):
    """
    Load the pretrained word2vec to a dict
    :param tmp_pretrained_word2vec:
    :return:
    """
    # don't use f.read,it will incur "MemoryError", ref: https://blog.csdn.net/u014159143/article/details/80360306
    # it is so slow to read the file with "f.readline()"
    # it is still so slow now!
    word2vec_list = []
    word2vec_path = tmp_pretrained_word2vec
    with open(word2vec_path,'r',encoding='utf-8') as f:
        word2vec_list = f.readlines()
        # word2vec_list.append(f.read())
    word2vec_list = [i.replace("\n","").rstrip().split(" ") for i in word2vec_list]

    word2vec_dict = {}
    word2vec_list_len = len(word2vec_list)
    for i in range(1,word2vec_list_len):
        word2vec_dict[word2vec_list[i][0]] = np.array([float(i) for i in word2vec_list[i][1:]])

    # print ("the number of words in the dict: ",len(word2vec_dict))

    # Notice: it is "string" in the list!!!!!
    # for i in range(len(word2vec_np[300])):
    #     print (i,":   ",word2vec_np[300][i])
    return word2vec_dict

def avgWord2Vec(tmp_corpus,tmp_pretrained_word2vec_path,using_tfidf=True):
    """
    Average the word2vec in the doc to get the doc2vec
    :param tmp_corpus:
    :param tmp_pretrained_word2vec_path:
    :param using_tfidf:
    :return:
    """
    print ('Average Word2Vecing ...')
    tmp_word2vec_dict = loadW2V(tmp_pretrained_word2vec_path)           # np.array
    docs_vec_list = []
    for i in range (len(tmp_corpus)):
        tmp_doc = tmp_corpus[i]
        tmp_doc_cut_list = tmp_doc.split(' ')
        tmp_doc_vecs = []
        tmp_tfidf = []
        keywords = jieba.analyse.extract_tags(tmp_doc, topK=len(tmp_doc_cut_list), withWeight=True)  # unusing topK
        # keywords_key = [i[0] for i in keywords]
        # keywords_value = [i[1] for i in keywords]
        keywords_dict = dict(keywords)
        for item in tmp_doc_cut_list:
            if item in tmp_word2vec_dict and item in keywords_dict:
                tmp_doc_vecs.append(tmp_word2vec_dict[item])            # TF Based
                tmp_tfidf.append(keywords_dict[item])
        # print ("document ",i,':\t',tmp_tfidf)
        if using_tfidf == False:
            docs_vec_list.append(np.mean(tmp_doc_vecs,axis=0))
        if using_tfidf == True:
            tmp_doc_vecs_tfidf = np.array(np.dot(np.matrix(tmp_doc_vecs).T,\
                                                 np.matrix(tmp_tfidf).T)).reshape((300,))           # Notice here!
            docs_vec_list.append(tmp_doc_vecs_tfidf)
    print ("AVG Word2Vec Shape: ",np.array(docs_vec_list).shape)
    return np.array(docs_vec_list)

###############################################SAVE VECTOR AND LABEL ###################################################

def saveVec(tmp_vecs,tmp_path,tmp_sparse=False):
    """
    Save the sparse or non-sparse martrix to a file
    :param tmp_vecs: a list of vec
    :param tmp_path: the path
    :param tmp_sparse: sparse or not
    :return: None
    """
    print("\nSaving vectors ...")
    if tmp_sparse==True:
        tmp_sparse_vec = sparse.csc_matrix(tmp_vecs)
        sparse.save_npz(tmp_path, tmp_sparse_vec)
    if tmp_sparse==False:
        np.save(tmp_path,tmp_vecs)
    print ("Saved Successfully!")
    return 0

def saveLabel(tmp_train_label,tmp_test_label,tmp_label_folder):
    """
    Save the label of training and test data to a folder
    :param tmp_train_label: a list of label
    :param tmp_test_label: a list of label
    :param tmp_label_folder: the folder
    :return: None
    """
    train_label_path = tmp_label_folder + 'train_label.npy'
    test_label_path = tmp_label_folder + 'test_label.npy'
    np.save(train_label_path,np.array(tmp_train_label))
    np.save(test_label_path, np.array(tmp_test_label))
    return 0

##################################################### MAIN ############################################################

def featureExtract(tmp_text,type_list,tmp_k_num =30,tmp_ngram_range =(1,1),\
                   tmp_pretrained_word2vec_path=pretrained_word2vec_path):
    """
    Feature Extraction
    :param tmp_text: the documents list
    :param type_list: choose which type of algorithms to be used.
    :param tmp_k_num: top K keywords in the process of count2vec or tfidf2vec
    :param tmp_ngram_range: for count2vec and tfidf2vec
    :param tmp_pretrained_word2vec_path:
    :return: save the vectors to a file
    """

    if 'count2vec' in type_list:
        count2vec_all = Count2Vec(text_all,tmp_ngram_range = tmp_ngram_range)
        count2vec_all_path = PreprocessedDataFolder + 'count2vec_' + str(tmp_k_num) + 'keywords.npz'
        saveVec(count2vec_all,count2vec_all_path,tmp_sparse=True)

    if 'tfidf2vec' in type_list:
        tfidf2vec_all = TFIDF2Vec(text_all,tmp_ngram_range = tmp_ngram_range)
        # tfidf2vec_fall = np.round(tfidf2vec_fall,6)                                   # for saving memory
        tfidf2vec_all_path = PreprocessedDataFolder + 'tfidf2vec_' + str(tmp_k_num) + 'keywords.npz'
        saveVec(tfidf2vec_all, tfidf2vec_all_path, tmp_sparse=True)

    if 'avg_word2vec' in type_list:
        avg_word2vec_all = avgWord2Vec(text_all,tmp_pretrained_word2vec_path=tmp_pretrained_word2vec_path)
        avg_word2vec_all_path = PreprocessedDataFolder + 'avg_word2vec.npy'
        saveVec(avg_word2vec_all, avg_word2vec_all_path, tmp_sparse=False)

    if 'avg_word2vec_tfidf' in type_list:
        avg_word2vec_all_tfidf = avgWord2Vec(text_all,\
                                       tmp_pretrained_word2vec_path=tmp_pretrained_word2vec_path,using_tfidf=True)
        avg_word2vec_all_tfidf_path = PreprocessedDataFolder + 'avg_word2vec_tfidf.npy'
        saveVec(avg_word2vec_all_tfidf, avg_word2vec_all_tfidf_path, tmp_sparse=False)
    return 0

if __name__ == '__main__':
    start_time = time.time()

    raw_train_data_path = RawDataFolder + 'raw_train/'
    raw_test_data_path = RawDataFolder + 'raw_test/'
    label_folder = PreprocessedDataFolder

    train_X,train_y,test_X,test_y = loadData(raw_train_data_path,raw_test_data_path)
    saveLabel(train_y, test_y, tmp_label_folder = label_folder)
    docsStat(train_y,test_y)

    k_num = 30                                                        # top K keywords for jieba.analyse.extract_tags
    train_X_preprocessed = textPreprocess(train_X)                    # train_X[0:10] for debugging
    test_X_preprocessed = textPreprocess(test_X)                      # test_X[0:10] for debugging
    text_all = train_X_preprocessed + test_X_preprocessed

    featureExtract(text_all,type_list=['avg_word2vec_tfidf'],tmp_k_num =30)

    end_time = time.time()
    print ("Time Used: ", end_time - start_time)