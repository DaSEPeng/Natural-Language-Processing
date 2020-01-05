"""
The First Lab of the Natural Language Processing Course (COMS0031132095)
- Name: Peng Li (李鹏)
- StudentID: 10175501102
- Date: 2019/09/12

- Project Structure                                                  # 代码架构
    - Load Data                                                      # 加载数据
    - Text Cleaning                                                  # 文本清洗
    - Sentence Segmentation                                          # 中英文分词
        - Extract English Words                                      # 提取英文词
            - Stem                                                   # 英文单词识别与形态还原
        - Extract Chinese Words                                      # 提取中文词汇
    - Calculate the Term Frequency (TF)                              # 词频统计
        - Calculate The Number of Each Word in A Document            # 统计每个词在文档中的个数
    - Calculate the Inverse Document Frequency (IDF)                 # 计算逆文档频率
    - Get the Term Frequency - Inverse Document Frequency (TF-IDF)   # 得到TF-IDF值
    - Get the TopK Key Words of Each Document                        # 得到每个文档的前K个关键词
    - Save the Result to a Txt File                                  # 将结果保存到一个TXT文档中
"""

# -*- coding: UTF-8 -*-
import jieba
import numpy as np
import pandas as pd
import os
import re
import time
import nltk

# Set the column width, ref: https://blog.csdn.net/weekdawn/article/details/81389865
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',200)

def loadData(path):
    """
    Load all the text in the file folder 'path' into a list
    :param path: the file folder containing all the text
    :return: a list containing 'file_name' and 'text'
    """
    files = os.listdir(path) # ref: https://ask.csdn.net/questions/684269
    tmp_text_list = []
    file_num = 0
    for file in files:
        base_file_name = os.path.basename(file)
        full_name = data_path + base_file_name
        with open (full_name, 'r',encoding = 'gb18030') as f:
            # 注意此处的编码方式，否则会报 illegal multibyte sequence 的错误
            # 参考：https://blog.csdn.net/jiasudu1234/article/details/71173281/
            text = f.read()
            tmp_text_list.append([base_file_name,text])
        file_num += 1
    print ("The Number of The Processed Files：",file_num)
    return tmp_text_list,file_num


def textClean(tmp_text_pd,file_num):
    """
    Clean the text data, just remove the numbers
    :param tmp_text_pd: the text data saved using pandas
    :param file_num: the number of documents
    :return: cleaned documents saved using pandas
    """
    # reg = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")

    print ("Text Cleaning：")
    print ("Before Cleaning：\n", tmp_text_pd[0:3])
    for i in range(file_num):
        for j in range(10):
            tmp_text_pd.iloc[i, 1] = tmp_text_pd.iloc[i, 1].replace(str(j),'') # 去除数字
    #   tmp_text_pd.iloc[i,1] = re.sub(reg, '',tmp_text_pd.iloc[i,1]) # 只保留英文大小写和数字
    print("After Cleaning:\n", tmp_text_pd[0:3])
    return tmp_text_pd


def stem(word):
    """
    Extract the stem of a English word using nltk
    :param word: a English word
    :return: the stem of the word
    """
    porter = nltk.PorterStemmer()
    return porter.stem(word)


def extractEnglish(str):
    """
    Extract the English words of a document
    :param str: the document
    :return: a list saving all the English words of the document
    """
    str_list = jieba.lcut(str)
    words_list = []
    zhPattern = re.compile('[\u4e00-\u9fa5]+')
    for i in range(len(str_list)):
        if zhPattern.search(str_list[i]) :
            continue
        else: # 如果不是中文（是英文）
            words_list.append(stem(str_list[i]))
    return words_list


def extractChinese(str):
    """
    Extract the Chinese words of a document
    :param str: the document
    :return: a list saving all the Chinese words of the document
    """
    reg = re.compile('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》（）？：:“”‘’！[\\]^_`{|}~]+')
    str = re.sub(reg, '', str)   # ref: https://www.jianshu.com/p/acf73e6f53a9
    str = str.replace('\n', '')
    str = str.replace('\t', '')
    str = "".join(str.split())
    # print (str)
    words_list = jieba.lcut(str)
    return words_list


def wordSegment(tmp_text_pd,file_num):
    """
    Sentence segmentation, extract the English words and Chinese words individually
    :param tmp_text_pd: all the documents
    :param file_num: the number of the documents
    :return: a DataFrame saving the words list of each document
    """
    print ("Word Segmentation：")
    print("Before Segmentation：\n", tmp_text_pd[0:3])
    for i in range(file_num):
        tmp_list = []
        # tmp_en_list = extractEnglish(tmp_text_pd.iloc[i,1])  # 提取英文词
        tmp_cn_list = extractChinese(tmp_text_pd.iloc[i,1])
        tmp_list.extend(tmp_cn_list)
        # tmp_list.extend(tmp_en_list)  # 将英文词加入所有词的List
        tmp_list = [i for i in tmp_list if len(i) > 1]  # 假设关键词的长度必须大于1
        tmp_text_pd.iloc[i, 1] = tmp_list
    print("After Segmentation:\n", tmp_text_pd[0:3])
    return tmp_text_pd


def countWT(tmp_text_list):
    """
    Count the number of each words in a document
    :param tmp_text_list: the document, a list
    :return: a dictionary whose key is the word and value is the number of the word
    """
    word_times_dict = {}
    tmp_list_len = len(tmp_text_list)
    for i in range(tmp_list_len):
        if tmp_text_list[i] not in word_times_dict:
            # https://blog.csdn.net/qq_19175749/article/details/79856010

            if tmp_text_list[i] not in words_dict_IDF:
                words_dict_IDF[tmp_text_list[i]] = 1
            else:
                words_dict_IDF[tmp_text_list[i]] += 1

            word_times_dict[tmp_text_list[i]] = 1  # 将这个词放入词典
            words_after_len = tmp_list_len-i-1      # 计算这个词后面的词的个数
            if words_after_len > 0:
                for j in range(1,words_after_len):
                    if tmp_text_list[i+j] == tmp_text_list[i]:
                        word_times_dict[tmp_text_list[i]] += 1
                        # tmp_text_list[i+j] = -1     # 注意这里是修改字符串还是赋值列表
                    else:
                        continue
            else:
                continue
        else:
            continue
    # print (word_times_dict)
    return word_times_dict


def calcuTF(tmp_text_pd,file_num):
    """
    Calculate the term-frequency of each document
    :param tmp_text_pd: all the documents, a list
    :param file_num: the number of the documents
    :return: a dictionary list saving the term-frequency of each document
    """
    dict_list = []
    # len_list = []
    for i in range(file_num):
        # print (i)
        tmp_word_times_dict = countWT(tmp_text_pd.iloc[i,1])
        tmp_len = sum(tmp_word_times_dict.values())
        # len_list.extend(tmp_len)
        for key,value in tmp_word_times_dict.items():
            tmp_word_times_dict[key] = value/tmp_len
        dict_list.append(tmp_word_times_dict)
    return dict_list


def calcuIDF(file_num):
    """
    Calculate the Inverse-Document-Frequency of each word
    :param file_num: the number of the documents
    :return: a dictionary which saving the Inverse-Document-Frequency of each word
    """
    for key, value in words_dict_IDF.items():
        if value > 0:  # 筛选至少在两个文档中出现的词，暂时没有筛选
            words_dict_IDF[key] = np.log(file_num/value) # 之前有保证这些词都在文档中有，暂时不用+1
        else:
            words_dict_IDF[key] = 0
    print ("\nWords Number of All Document：",len(words_dict_IDF))

    # print ("\nWord List：")
    # i = 0
    # for key,value in words_dict_IDF.items():
    #     print (i, '\t', key)
    #     i += 1
    # print (words_dict_IDF)

    return words_dict_IDF


def calcuTF_IDF(tmp_TF_dict_list,tmp_words_idf_dict):
    """
    Calculate the TF-IDF of each term of each document
    :param tmp_TF_dict_list: the TF dictionary list of each document
    :param tmp_words_idf_dict: the IDF dictionary list saving the IDF value of each term
    :return: the TF-IDF list of each document
    """
    tmp_tf_idf_list = []
    list_len = len(tmp_TF_dict_list)
    for i in range(list_len):
        tmp_dict = {}
        for key,value in tmp_TF_dict_list[i].items():
            # if key in tmp_tf_idf_list:  # 为了配合TF-IDF筛选
            tmp_dict[key] = value * tmp_words_idf_dict[key]
        tmp_tf_idf_list.append(tmp_dict)
    return tmp_tf_idf_list


def topK(tmp_tf_idf_list,k):
    """
    Sort the TF-IDF value and get the top K key words of each document
    :param tmp_tf_idf_list: the TF-IDF list of each document
    :param k: the K
    :return: a list containing the top K key words of each document
    """
    list_len = len(tmp_tf_idf_list)
    tmp_topK_list = []

    k = k if k < list_len else list_len  # 如果列表中‘词’的数目小于K就输出整个列表

    for i in range(list_len):
        # ref: https://www.cnblogs.com/yoyoketang/p/9147052.html
        sorted_tuple_list = sorted(tmp_tf_idf_list[i].items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_tuple_list[:k]
        tmp_topK_list.append(top_k)
    return tmp_topK_list


def saveResult(path, tmp_file_name, tmp_topK, file_num):
    """
    Save the result to a txt file
    :param path: the path of the txt file
    :param tmp_file_name: the name list of the documents
    :param tmp_topK: the topK list of each document
    :param file_num: the number of the documents
    :return: the txt file containing the name and top K terms of each document
    """
    with open(path,"w") as f:
        for i in range(file_num):
            file_name = tmp_file_name[i]
            tmp_file_topK = tmp_topK[i]
            newline = file_name + '\t'+str(tmp_file_topK)
            f.writelines(newline+"\n")
    print ("\nSave Successfully!")
    return 0


def keywordsExtract(input_path,output_path,tmp_k):
    """
    Main function, extract the topK keywords of the documents and save them to a txt file
    :param input_path: the path of the folder containing the documents
    :param output_path: the path of the output file
    :param tmp_k: the number of K
    :return: 0
    """
    # load data
    text,file_number = loadData(input_path)
    text_pd = pd.DataFrame(text)
    text_pd.columns = ['file_name','text']

    # text clean and word segmentation
    text_pd_cleaned = textClean(text_pd,file_number)
    text_pd_seged = wordSegment(text_pd_cleaned,file_number)

    # calculate the TF
    TF_dict_list = calcuTF(text_pd_seged,file_number)
    print ("\nThe TF Values of The First Three Documents：")
    print (TF_dict_list[0:3])

    # calculate the IDF and get the TF-IDF
    words_idf_dict = calcuIDF(file_number)
    tf_idf_list = calcuTF_IDF(TF_dict_list, words_idf_dict)
    print ("\nThe TF-IDF Values of The First Three Documents：")
    print (tf_idf_list[0:3])

    # get the topK key words of each document

    topK_list = topK(tf_idf_list,tmp_k)
    test_num = 10
    print ("\nThe Top ",tmp_k," Keywords and Corresponding TF-IDF Values of The First ",test_num, "Documents: ")
    for i in range (test_num):
        print (topK_list[i])

    # save the result to a txt file
    saveResult(output_path,text_pd['file_name'],topK_list,file_number)
    return 0



if __name__ == '__main__':

    start_time = time.time()

    data_path = 'data/'  # 数据文件夹路径
    result_path = 'result.txt'  # 保存结果的文档

    words_dict_IDF = {}  # 保存不同的词出现在不同文档的次数

    K = 10
    keywordsExtract(data_path,result_path,tmp_k=K)

    end_time = time.time()
    time_used = end_time - start_time
    print ("\nTime Used: ", time_used)