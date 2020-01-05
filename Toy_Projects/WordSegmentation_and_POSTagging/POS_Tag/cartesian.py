#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 用python实现求笛卡尔积

import itertools

class cartesian(object):
    def __init__(self):
        self._data_list=[]

    def add_data(self,data=[]): #添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self): #计算笛卡尔积
        tmp_list = []
        for item in itertools.product(*self._data_list):
            tmp_list.append(item)
        return tmp_list