"""
基于word2vec + kmeans 和 count2vec + LDA 模型的文本聚类，并利用外部指标进行FMI与ARI结果评测

author: Peng Lee
date: 2019/11/11
contact: ruhao9805@163.com
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import pickle
# https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index
from sklearn.metrics import fowlkes_mallows_score,adjusted_rand_score
from scipy import sparse
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation,NMF


data_folder = 'data/preprocessed_data/'
avgW2V_data_path = data_folder + 'avg_word2vec.npy'
count2vec_data_path = data_folder + 'count2vec_30keywords.npz'
tfidf2vec_data_path = data_folder + 'tfidf2vec_30keywords.npz'
train_label_path = data_folder + 'train_label.npy'
test_label_path = data_folder + 'test_label.npy'
model_path = 'kmeans.pickle'


############################################ 加载数据集 ###############################################

def loadData(dataset='avg_word2vec'):
    """
    加载训练集，测试集，训练集标签，测试集标签
    :return:
    """
    if dataset == 'avg_word2vec':
        tmp_all_data = np.load(avgW2V_data_path)
    if dataset == 'count2vec':
        tmp_all_data = sparse.load_npz(count2vec_data_path).toarray()
    if dataset == 'tfidf2vec':
        tmp_all_data = sparse.load_npz(tfidf2vec_data_path).toarray()

    tmp_training_data = tmp_all_data[:9804]
    tmp_testing_data = tmp_all_data[9804:]

    print ("Training Data Shape: ", tmp_training_data.shape,"; Testing Data Shape: ", tmp_testing_data.shape)

    tmp_training_label = np.load(train_label_path)
    tmp_testing_label = np.load(test_label_path)
    return tmp_training_data,tmp_testing_data,tmp_training_label,tmp_testing_label

#################################### 利用FMI与ARI进行评测 ############################################

def evaluate(tmp_pred,tmp_true,tmp_n = 20):
    """
    使用sklearn提供的fowlkes_mallows_score()和adjusted_rand_score()函数对聚类结果进行评测
    :param tmp_pred:
    :param tmp_true:
    :param tmp_n:
    :return:
    """
    fmi_score = fowlkes_mallows_score(tmp_true, tmp_pred)  ## FMI值
    print('数据聚%d类FMI评价分值为：%f' % (tmp_n, fmi_score))
    ri_score = adjusted_rand_score(tmp_true, tmp_pred)  ## 调整过的RI值
    print('数据聚%d类ARI评价分值为：%f' % (tmp_n, ri_score))
    return 0

########################################### KMEANS 聚类算法 ##########################################

def kmeansCluster(tmp_data,dataset_label,scoring = True,if_tsne = False,model_saved=False):
    """
    利用Kmeans进行聚类，可以选择保存模型或者将结果可视化
    :param tmp_data:
    :param dataset_label:
    :param scoring:
    :param if_tsne:
    :param model_saved:
    :return:
    """
    tmp_n_clusters = 20
    ## 如果已经保存过模型，则直接调用；如果没有保存过，则再次训练
    if model_saved == True:
        with open(model_path, 'rb') as f:
            tmp_kmeans = pickle.load(f)
    else:
        tmp_kmeans = KMeans(n_clusters=tmp_n_clusters, random_state=123,n_init=10,max_iter=300).fit(tmp_data)

    ## 打印模型参数
    print ('MODEL：',tmp_kmeans)
    ## 打印模型前几项预测结果
    print ('Pred Labels: ', tmp_kmeans.labels_[0:40])
    print ('True Labels: ', dataset_label[0:40])

    ## 评价结果
    if scoring == True:
        evaluate(dataset_label, tmp_kmeans.labels_, tmp_n=tmp_n_clusters)

    ## 通过TSNE进行可视化
    if if_tsne == True:
        tsne = TSNE(n_components=2, init='random', random_state=177).fit(tmp_data)
        df = pd.DataFrame(tsne.embedding_)
        df['labels'] = tmp_kmeans.labels_
        df_list = []
        for i in range (20):
            df_list.append(df[df['labels'] == i])
        fig = plt.figure(figsize=(9, 6))
        style_list = ['r+','bo','m*','kx','rs','bd','mp','kh','r<','b>', \
                      'g+', 'co', 'y*', 'rx', 'gs', 'cd', 'yp', 'rh', 'g<', 'c>']  ## 设置图像格式

        for i in range(20):
            plt.plot(df_list[i][0],df_list[i][1],style_list[i])                    ## 绘制图像
        plt.savefig('ClusterResult.png')
        plt.show()

    ## 如果没有保存过模型，则保存模型
    if model_saved == False:
        with open(model_path, 'wb') as f:
            pickle.dump(tmp_kmeans, f)

    return tmp_kmeans

################################################## LDA 主题模型 ###########################################

def LDA(tmp_data,dataset_label):
    # tmp_doc_topic_pri = np.array([48, 40, 36, 34, 26, 26, 20, 18, 12, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]) / 275

    tmp_n_components=20
    lda = LatentDirichletAllocation(n_components=tmp_n_components,random_state=233,doc_topic_prior=0.0001)
    print ("MODEL: ",lda)
    docres = lda.fit_transform(tmp_data)

    pred_labels = np.argmax(docres,axis=1)
    print ("LDA主题模型预测标签：", pred_labels[0:40])
    print('真实的标签: ', dataset_label[0:40])

    evaluate(dataset_label, pred_labels, tmp_n=tmp_n_components)
    return docres

############################################### 主函数 #################################################

if __name__ == '__main__':
    start_time = time.time()

    model = 'lda'  ## lda or kmeans

    if model == 'lda':
        training_data,testing_data,training_label,testing_label = loadData('count2vec')
        print ("训练集聚类结果：")
        lda_result = LDA(training_data, training_label)
        print("测试集聚类结果：")
        lda_result = LDA(testing_data, testing_label)
    if model == 'kmeans':
        training_data, testing_data, training_label, testing_label = loadData('avg_word2vec')
        # all_data = np.concatenate((training_data,testing_data),axis=0)
        # scaler = preprocessing.StandardScaler().fit(training_data)
        # scaler = scaler.transform(training_data)
        print("训练集聚类结果：")
        kmeansCluster(training_data,\
                      dataset_label = training_label, scoring = True,if_tsne= False,model_saved=False)
        print("测试集聚类结果：")
        kmeansCluster(testing_data,\
                      dataset_label = testing_label, scoring = True,if_tsne= False,model_saved=False)


    end_time = time.time()
    print ("Time Used: ", end_time - start_time)