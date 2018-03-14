#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018/3/14 0014 11:09 
# Author: Lyu

# 优点：数据较少时，仍然有效，可以处理多分类问题
# 缺点：对于输入数据的方式较为敏感
# 适用数据类型：标称型数据

import numpy as np

def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1] # 1:代表侮辱性文字，0:代表正常言论
    return posting_list, class_vec

def create_vocab_list(dataset):
    vocab_set = set()
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def set_of_word2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word %s not in my vocabulary !" % word
    return return_vec

def train_nb(train_X, train_y):
    num_train_docs = len(train_X)
    num_words = len(train_X[0])
    p_abusive = sum(train_y)/np.float(num_train_docs) # 侮辱性文字出现的概率
    word_num_0 = np.zeros(num_words); word_num_1 = np.zeros(num_words) # 初始化各个词在侮辱和正常言论中出现的次数
    # todo 不理解这里为什么要把代表侮辱和正常文章的词汇数累加起来
    num_0 = 
    for i in range(num_train_docs):
        if train_y[i] == 1:
            word_num_1 += train_X[i]




if __name__ == '__main__':
    post_list,  class_list= load_dataset()
    vocab_list = create_vocab_list(post_list)
    print set_of_word2vec(vocab_list, post_list[0])

