#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: DaSEPeng
@date: 2020/01/01
@description: 论文“《基于路径与深度的同义词词林词语相似度计算》陈宏朝等”的实现
代码主要参考 https://github.com/yaleimeng/Final_word_Similarity 的实现，
修复了其中的BUG，使代码可以运行，并重整了整个代码提高了模块化与可读性
'''

class CilinSimilarity(object):

    def __init__(self,cilin_path,alpha,weights):
        """
        初始化
        :param cilin_path: 数据集路径
        :param alpha: alpha的值，论文中是0.9
        :param weights: 各边权重的值，论文中是 [0.5, 1.5, 4, 6, 8]
        """
        self.N = 0                              # 词林单词数（含重）
        self.code_word = {}                     # {编码：单词list}
        self.word_code = {}                     # {单词：编码list}
        self.vocab = set()                      # 词林单词数（去重）
        self.cilin_path = cilin_path
        self.alpha = alpha
        self.weights = weights
        self.read_cilin(self.cilin_path)

    def read_cilin(self,tmp_cilin_path):
        """
        读入数据
        :param tmp_cilin_path: 词林保存的路径
        :return: 词林数据集以及多个初始化数据
        """
        with open(tmp_cilin_path, 'r', encoding='gbk') as f:
            for line in f.readlines():
                res = line.split()
                code = res[0]
                words = res[1:]
                self.vocab.update(words)
                self.N += len(words)

                if len(code)==8:
                    self.code_word[code] = words                      # {编码：单词list}
                    for w in words:
                        if w in self.word_code.keys():                # {单词：编码list}
                            self.word_code[w].append(code)
                        else:
                            self.word_code[w] = [code]

    def get_common_str(self, c1, c2):
        """
        返回编码的公共前缀，但是这里需要注意有的层是2位数字
        :param c1: 第一个编码
        :param c2: 第二个编码
        :return: 公共前缀
        """
        res = ''
        for i, j in zip(c1, c2):
            if i == j:
                res += i
            else:
                break
        if 3 == len(res) or 6 == len(res):
            res = res[:-1]
        return res

    def get_LSP_below_layer(self, common_str):
        """
        返回公共父节点的下一层的层数，对应原论文图1中从下到上，1-6层（第6层不用）
        :param common_str: 公共前缀编码
        :return: 公共父节点的下一层的层数
        """
        length = len(common_str)
        table = {1: 4, 2: 3, 4: 2, 5: 1, 7: 0}             # 没有公共节点的话，LSP_below设为第0层
        if length in table.keys():
            return table[length]
        return 5

    def code_split(self, c):
        """
        将编码拆分成各层编码
        :param c: 编码
        :return: 各层编码组成的列表
        """
        return [c[0], c[1], c[2:4], c[4], c[5:7], c[7]]

    def get_K(self, c1, c2):
        """
        返回两个编码对应在公共父节点的分支中的距离，即原论文中的 K
        :param c1: 第一个编码
        :param c2: 第二个编码
        :return: 两个编码对应在公共父节点的分支中的距离
        """
        if c1[0] != c2[0]:
            return abs(ord(c1[0]) - ord(c2[0])) # 第一层编号，一个大写字母
        elif c1[1] != c2[1]:
            return abs(ord(c1[1]) - ord(c2[1])) # 第二层编号，一个小写字母
        elif c1[2] != c2[2]:
            return abs(int(c1[2]) - int(c2[2])) # 第三层编码，一个二位数字
        elif c1[3] != c2[3]:
            return abs(ord(c1[3]) - ord(c2[3])) # 第四层编码，一个大写字母
        else:
            return abs(int(c1[4]) - int(c2[4])) # 最后一层编号，一个二位数字

    def get_N(self, common_str):
        """
        计算公共父节点的分支数，即原论文中的 N
        :param common_str: 公共前缀编码
        :return: 公共父节点的分支数
        """
        if not common_str:
            return 14                                             # 如果没有公共子串（空字符），则第五层有14个大类。

        siblings = set()                                           # 兄弟节点集合
        LSP_below_layer = self.get_LSP_below_layer(common_str)
        for c in self.code_word.keys():
            if c.startswith(common_str):
                c_split = self.code_split(c)
                siblings.add(c_split[5 - LSP_below_layer])
        return len(siblings)

    def sim_by_code(self, c1, c2):
        """
        计算两个编码的相似度
        :param c1: 第一个编码
        :param c2: 第二个编码
        :return: 两个编码的相似度
        """
        c1_split, c2_split = self.code_split(c1), self.code_split(c2)
        common_str = self.get_common_str(c1, c2)
        LSP_below_layer = self.get_LSP_below_layer(common_str)
        LSP_layer = LSP_below_layer + 1

        if len(common_str) >= 7:
            if common_str[-1] == '=':
                return 1
            elif common_str[-1] == '#':
                return 0.5

        K = self.get_K(c1_split, c2_split)
        N = self.get_N(common_str)
        weights = [0]
        weights.extend(self.weights)                          # 为了与层数对应，第一个元素设置为 0
        Depth = 0.9 if LSP_layer > 5 else sum(weights[LSP_layer:])
        Path = 2 * sum(weights[:LSP_layer])
        alpha = self.alpha
        beta = K / N * weights[LSP_below_layer]
        return (Depth + alpha) / (Depth + alpha + Path + beta) if LSP_layer <= 5 else alpha / (alpha + Path + beta)

    def sim(self, w1, w2):
        """
        计算两个词的相似度
        :param w1: 第一个词
        :param w2: 第二个词
        :return: 词的相似度
        """
        # 如果有一个词不在词林中，则相似度为0
        if w1 not in self.vocab or w2 not in self.vocab:
            return 0

        # 获取两个词的编码
        code1 = self.word_code[w1]
        code2 = self.word_code[w2]

        # 选取相似度最大值
        sim_max = 0
        for c1 in code1:
            for c2 in code2:
                cur_sim = self.sim_by_code(c1, c2)
                sim_max = cur_sim if cur_sim > sim_max else sim_max
        return sim_max