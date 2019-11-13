# -*- coding: utf-8 -*-

import pandas as pd                 # 0.25.2
import numpy as np
from math import log
import cartesian
import time

pos_transition = [
[80000, 10000 , 12000 , 80000 , 5000  , 10000 ],
[30000, 1000  ,  2000  ,  20000 ,  10000,  5000   ],
[40000, 1000 ,  500   ,  5000 , 5000 , 10000  ],
[50000, 5000  ,  5000 ,  20000 ,  4000  , 10000  ],
[30000, 8000 ,  4000  ,  7000  ,  1000,  20000  ],
[20000 ,10000 , 6000 , 30000 , 5000 ,  9000   ]]

pos_fre = [
['n', 200000],
['c', 100000],
['p', 100000],
['v', 200000],
['a', 100000],
['d', 100000]]

# 注意不能多打空格
word_fre = [
['人民' , 'n' ,5000],
['水平', 'n', 4000],
['收入', 'n', 4000],
['水平', 'a'  , 1000 ],
['和', 'c'  ,2000],
['进一步','n',1000 ],
['和', 'p'  , 1000],
['进一步', 'd' , 2000],
['和', 'v'  , 200],
['提高', 'n'  , 1000 ],
['生活', 'n'  , 5000],
['提高', 'v'  , 4000 ],
['生活', 'v'  , 2000 ]]

def main():
    pos_transition_pd = pd.DataFrame(pos_transition)
    pos_transition_pd.columns = ['n','c','p','v','a','d']
    pos_transition_pd.index = ['n','c','p','v','a','d']
    # print (pos_transition_pd)

    pos_fre_pd = pd.DataFrame(pos_fre)
    pos_fre_pd.columns = ['pos','fre']
    # print(pos_fre_pd)

    word_fre_pd = pd.DataFrame(word_fre)
    word_fre_pd.columns = ['word','pos','fre']
    # print(word_fre_pd)

    def TransitionPro(a, b):
        # 计算转移概率 P(pos_b|pos_a)
        p_cf = pos_transition_pd.loc[a,b]
        p_c = pos_fre_pd.loc[lambda df:df.pos==a,'fre'].to_numpy()[0]
        # print ("Transition Pro: ", p_cf/p_c)
        return p_cf/p_c

    def OutputPro(tmp_pos,tmp_word):
        # 计算输出概率
        tmp_pos_fre = pos_fre_pd.loc[pos_fre_pd['pos'] == tmp_pos, 'fre'].to_numpy()[0]       # .to_numpy()!!!
        tmp_pos_and_word = word_fre_pd.loc[(word_fre_pd['word'] \
                                            == tmp_word)&(word_fre_pd['pos'] ==tmp_pos),'fre'].to_numpy()[0]
        # print ("Output Pro: ", tmp_pos_and_word/tmp_pos_fre)
        return tmp_pos_and_word/tmp_pos_fre

    def allPos(tmp_word):
        # 输出某个单词所有可能的词性
        return [i for i in word_fre_pd.loc[(word_fre_pd['word'] == tmp_word),'pos']]

    def calPro(word_list,pos_list):
        # 计算某段文字对应标注是pos_list的概率
        tmp_pro = 0
        list_len = len(word_list)
        for i in range (list_len):
            # print ("#"*100,": ",tmp_pro)
            if i < list_len-1:
                tmp_pro = tmp_pro- log(OutputPro(pos_list[i],word_list[i])) \
                          - log(TransitionPro(pos_list[i], pos_list[i+1]))            # -np.log()
            else:
                tmp_pro = tmp_pro - log(OutputPro(pos_list[i],word_list[i]))
        return tmp_pro

    def HMM(tmp_sentence_list):
        # HMM算法，没有使用Viterbi
        start_time = time.time()
        all_route = cartesian.cartesian()
        for word in tmp_sentence_list:
            all_route.add_data(list(allPos(word)))
        all_route = all_route.build()
        all_route_value = [calPro(tmp_sentence_list,i) for i in all_route]
        print ("HMM result: ", all_route[np.argmax(all_route_value)])
        end_time = time.time()
        print ("Time Used: ", end_time - start_time)
        return 0

    def HMM_Viterbi(tmp_sentence_list):
        # 基于Viterbi的HMM算法
        start_time = time.time()

        best_tag_list_reverse = []                     # 记录最好的标签序列
        list_len = len(tmp_sentence_list)

        best_value_per_pos_per_word = []       # 每个词每个词性的最优值以及它前面对应的词、词的词性

        for i  in range(list_len):
            # print ("处理第",i+1,"个词中")
            tmp_word = tmp_sentence_list[i]
            if i != 0:
                last_word = tmp_sentence_list[i-1]

            tmp_all_pos = allPos(tmp_word)

            for pos in tmp_all_pos:
                if i == 0:
                    tmp_sigma = 0 - log(OutputPro(pos,tmp_word))                 # 当前词的分数
                    best_value_per_pos_per_word.append([tmp_word,pos,tmp_sigma,'',''])
                    best_value_per_pos_per_word_pd = pd.DataFrame(best_value_per_pos_per_word)  # 转成DataFrame格式
                    best_value_per_pos_per_word_pd.columns = ['word','pos','best_value','last_word','last_pos']
                if i != 0:
                    last_word_pos_list = allPos(last_word)
                    tmp_pos_sigma_list = []                           # 暂时记录当前词性对应的前面的词性以及自己的sigma值
                    for last_pos in last_word_pos_list:
                        sigma_last = best_value_per_pos_per_word_pd.loc[(best_value_per_pos_per_word_pd.word==last_word)\
                            &(best_value_per_pos_per_word_pd.pos == last_pos),'best_value'].to_numpy()[0]
                        tmp_sigma = sigma_last - log(TransitionPro(last_pos,pos)) - log(OutputPro(pos,tmp_word))
                        tmp_pos_sigma_list.append([last_pos,tmp_sigma])

                    max_value = 0

                    for j in range(len(tmp_pos_sigma_list)):                      # 选择当前词性最大值对应的前面的词性
                        if tmp_pos_sigma_list[j][1] > max_value:
                            max_value = tmp_pos_sigma_list[j][1]
                            max_value_last_pos = tmp_pos_sigma_list[j][0]

                    # 下面的代码有点难看
                    best_value_per_pos_per_word.append([tmp_word, pos, max_value,last_word,max_value_last_pos])
                    best_value_per_pos_per_word_pd = pd.DataFrame(best_value_per_pos_per_word)  # 转成DataFrame格式
                    best_value_per_pos_per_word_pd.columns = ['word','pos','best_value','last_word','last_pos']
            # if i == list_len-1:
            #     print ("最后的处理结果：\n ",best_value_per_pos_per_word_pd)

        lastest_word = tmp_sentence_list[-1]
        lastest_pos_index = best_value_per_pos_per_word_pd.loc[best_value_per_pos_per_word_pd.word==lastest_word,\
                                                      'best_value'].idxmax()

        tmp_index = lastest_pos_index
        lastest_pos = best_value_per_pos_per_word_pd.loc[lastest_pos_index,'pos']
        best_tag_list_reverse.append(lastest_pos)
        for i in range(list_len-1,0,-1):
            pre_word = tmp_sentence_list[i-1]                       # 之前的一个词
            pre_word_pos = best_value_per_pos_per_word_pd.loc[tmp_index,'last_pos']
            tmp_index = best_value_per_pos_per_word_pd.loc[(best_value_per_pos_per_word_pd.word==pre_word)\
                                                           &(best_value_per_pos_per_word_pd.pos==pre_word_pos)\
                ,'best_value'].idxmax()
            best_tag_list_reverse.append(pre_word_pos)
        best_tag_list_reverse

        best_tag_list = []
        for m in range(len(best_tag_list_reverse)-1,-1,-1):
            best_tag_list.append(best_tag_list_reverse[m])

        print("HMM_Viterbi result: ",best_tag_list)
        end_time = time.time()
        print("Time Used: ", end_time - start_time)
        return 0

    return HMM,HMM_Viterbi

if __name__ == '__main__':
    sent_list = ['人民', '收入', '和', '生活', '水平', '进一步', '提高']
    print("Processing ", sent_list)

    HMM,HMM_Viterbi = main()

    HMM(sent_list)
    HMM_Viterbi(sent_list)

