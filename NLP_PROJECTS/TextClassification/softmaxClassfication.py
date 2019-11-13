"""
Softmax Classification, train_test_split,PCA,Oversampling,Sparse saving, micro vs macro included.
author: Peng Lee
date: 2019/10/22
contact: ruhao9805@163.com
"""

import numpy as np
import pandas as pd
# https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/linear_model/logistic.py#L1202
from sklearn.linear_model import LogisticRegression                              # version: 0.21.3
from scipy import sparse
from sklearn.decomposition import PCA
import time
# https://blog.csdn.net/kizgel/article/details/78553009?locationNum=6&fps=1#imblearn-package-study
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN                # version: 0.5.0
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

PreProcessedDataFolder = 'data/preprocessed_data/'

############################################ LOAD VECTOR ############################################################

def loadVec(tmp_type,pca=False,n_components = 300,tmp_over_sampling=False):
    """
    Load the Vectors
    :param tmp_type:
    :param pca:
    :param n_components: n_components must be between 1 and n_features=300 with svd_solver='randomized'
    :param tmp_over_sampling:
    :return:
    """
    tmp_train_y = np.load(PreProcessedDataFolder + 'train_label.npy')  # Len: 9804
    tmp_test_y = np.load(PreProcessedDataFolder + 'test_label.npy')  # Len: 9833

    if tmp_type == 'count2vec':
        tmp_vec = sparse.load_npz(PreProcessedDataFolder + 'count2vec_30keywords.npz').toarray()  # (19637, 62183)
    if tmp_type == 'tfidf2vec':
        tmp_vec = sparse.load_npz(PreProcessedDataFolder + 'tfidf2vec_30keywords.npz').toarray()  # (19637, 62183)
    if tmp_type == 'avg_word2vec':
        tmp_vec = np.load(PreProcessedDataFolder + 'avg_word2vec.npy')                            # (19637, 300)
    print("Original Shape: ", tmp_vec.shape)

    if pca == True:                                                   # MemoryError
        pca = PCA(n_components = n_components,svd_solver='randomized') # https://www.cnblogs.com/pinard/p/6243025.html
        pca.fit(tmp_vec)
        tmp_vec = pca.transform(tmp_vec)
        print("Shape After PCA: ",tmp_vec.shape)

    tmp_train_X = tmp_vec[:len(tmp_train_y)]
    tmp_test_X = tmp_vec[len(tmp_train_y):]

    if tmp_over_sampling == True:
        # tmp_train_X, tmp_train_y = RandomOverSampler().fit_sample(tmp_train_X, tmp_train_y)
        tmp_train_X, tmp_train_y = SMOTE().fit_sample(tmp_train_X, tmp_train_y)
        print ("Shape After OverSampling: ", tmp_train_X.shape)
    return tmp_train_X,tmp_train_y,tmp_test_X,tmp_test_y

############################################### EVALUATE #############################################################

def evaluate(true,predicted):
    """
    Evaluate the result
    :param true:
    :param predicted:
    :return:
    """
    print ("Evaluation Result: ")
    tmp_precision_score_micro = precision_score(true,predicted,average='micro')
    tmp_recall_score_micro = recall_score(true, predicted, average='micro')
    tmp_f1_score_micro = f1_score(true,predicted,average='micro')
    tmp_precision_score_macro = precision_score(true,predicted,average='macro')
    tmp_recall_score_macro = recall_score(true, predicted, average='macro')
    tmp_f1_score_macro = f1_score(true,predicted,average='macro')
    print ("micro precision: ",tmp_precision_score_micro)
    print("micro recall: ", tmp_recall_score_micro)
    print ("micro_f1: ",tmp_f1_score_micro)
    print("macro_precision: ", tmp_precision_score_macro)
    print("macro_recall: ",tmp_recall_score_macro)
    print("macro_f1: ",tmp_f1_score_macro)
    print ("\n")
    return tmp_precision_score_micro,tmp_recall_score_micro,\
           tmp_f1_score_micro,tmp_precision_score_macro,\
           tmp_recall_score_macro,tmp_f1_score_macro


############################################## PLOT VALIDATION RESULT ###############################################

def plotValidResult():
    """
    Plot Validation Result
    :return:
    """
    result = np.load(PreProcessedDataFolder + 'valid_result.npy')
    result_pd = pd.DataFrame(result)
    result_pd.columns = [['iter','train_p_micro','train_r_micro','train_f1_micro', \
                          'train_p_macro','train_r_macro','train_f1_macro', \
                            'valid_p_micro', 'valid_r_micro','valid_f1_micro', \
                            'valid_p_macro', 'valid_r_macro', 'valid_f1_macro']]
    result_pd.index = result_pd.iter
    fig, ax = plt.subplots()
    ax.plot(result_pd[['train_f1_micro','train_f1_macro','valid_f1_micro','valid_f1_macro']].iloc[4:,:])
    ax.set_ylabel('value')
    ax.set_xlabel('iter')
    ax.set_title('Validation Result')
    ax.legend(labels = ['train_f1_micro','train_f1_macro','valid_f1_micro','valid_f1_macro'])
    plt.show()
    fig.savefig('valid_result.eps', dpi=600, format='eps')
    return 0


####################################################### MAIN #########################################################

def main(train_X,train_y,test_X,test_y,valid=False,max_iter=100):
    """
    Main Function
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :param valid: valid only, used to choose the best para
    :param max_iter: default max_iter if valid == False
    :return:
    """
    if valid == True:
        train_X_splited, valid_X_splited, train_y_splited, valid_y_splited = \
            train_test_split(train_X, train_y, test_size=0.33, random_state=666)  # over_sampling is needed!
        result_list = []
        for iter in range (0,1001,25):
            print ("Iter No.",iter)
            lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=iter)
            lr_clf.fit(train_X_splited, train_y_splited)
            train_splited_predict = lr_clf.predict(train_X_splited)
            print ("Evaluate the train data: ")
            train_p_micro,train_r_micro,\
            train_f1_micro,train_p_macro,train_r_macro,train_f1_macro = evaluate(train_y_splited, train_splited_predict)
            valid_splited_predict = lr_clf.predict(valid_X_splited)
            print("Evaluate the valid data: ")
            valid_p_micro, valid_r_micro, \
            valid_f1_micro, valid_p_macro, valid_r_macro, valid_f1_macro = evaluate(valid_y_splited, valid_splited_predict)
            result_list.append([iter,\
                                train_p_micro,train_r_micro,train_f1_micro,\
                                train_p_macro,train_r_macro,train_f1_macro, \
                                valid_p_micro, valid_r_micro,valid_f1_micro, \
                                valid_p_macro, valid_r_macro, valid_f1_macro])
        np.save(PreProcessedDataFolder + 'valid_result.npy',np.array(result_list))
        plotValidResult()
    else:
        # C:smaller values specify stronger regularization
        lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=max_iter)
        lr_clf.fit(train_X, train_y)
        train_predict = lr_clf.predict(train_X)
        print("Evaluate the train data: ")
        evaluate(train_y, train_predict)
        test_predict = lr_clf.predict(test_X)
        print("Evaluate the test data: ")
        evaluate(test_y, test_predict)
    print ("END!")
    return 0

if __name__ == '__main__':
    start_time = time.time()
    train_X,train_y,test_X,test_y = loadVec('avg_word2vec',pca=False,tmp_over_sampling=True)  # type np.array
    main(train_X, train_y, test_X, test_y,valid=False,max_iter=625)
    end_time = time.time()
    print ("Time Used: ", end_time - start_time)



