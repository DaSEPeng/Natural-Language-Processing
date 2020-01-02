"""
测试cilin.py的代码的运行结果，并与现在常用的基于Word2vec计算词义相似度的方式做对比

"""
import pandas as pd
from cilin import CilinSimilarity
from word2vec import Word2vecSimilarity

# 读取测试数据
test_path = 'cilin_test.csv'
test = pd.read_csv(test_path,encoding='gbk',header=None)
test.columns = ['word1','word2']

## 词林计算相似度
alpha=0.9
weights = [0.5, 1.5, 4, 6, 8]
cs = CilinSimilarity(cilin_path = './cilin_extend.txt',alpha=alpha,weights=weights)

## 词向量计算相似度
# w2v = Word2vecSimilarity()

cilin_sim_result = []
# w2v_cos_sim = []
# w2v_eu_sim = []

for i in range(len(test)):
    word1 = test['word1'][i]
    word2 = test['word2'][i]
    cilin_sim = cs.sim(word1,word2)
    cilin_sim_result.append(cilin_sim)
    print (word1,word2,cilin_sim)
    # w2v_cos = w2v.w2v_sim(word1,word2,'cos')
    # w2v_eu = w2v.w2v_sim(word1,word2,'eu')
    # w2v_cos_sim.append(round(w2v_cos),4)
    # w2v_eu_sim.append(round(w2v_eu),4)

#
# file=open('w2v_cos_sim.txt','w')
# file.write(str(w2v_cos_sim))
# file.close()
#
# file=open('w2v_eu_sim.txt','w')
# file.write(str(w2v_eu_sim))
# file.close()
#


