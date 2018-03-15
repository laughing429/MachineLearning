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
    """
    词集模型，特征只反应词是否出现在文档中
    :param vocab_list:
    :param input_set:
    :return:
    """
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word %s not in my vocabulary !" % word
    return return_vec

def bag_of_word2vec(vocab_list, input_set):
    """
    词袋模型，特征反应的是词在文档中出现了几次
    :return:
    """
    return_vec = [0]*len(vocab_list)
    for word in vocab_list:
        if word in input_set:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb(train_X, train_y):
    num_train_docs = len(train_X)
    num_words = len(train_X[0])
    p_abusive = sum(train_y)/np.float(num_train_docs) # 侮辱性文字出现的概率
    # word_num_0 = np.zeros(num_words); word_num_1 = np.zeros(num_words) # 初始化各个词在侮辱和正常言论中出现的次数
    # num_0 = 0.0
    # num_1 = 0.0
    # 利用贝叶斯分类器对文档进行分类的时候，要计算多个概率的乘积，如果其中一个概率为0，那么最后的乘积也为0。为了避免这个情况
    word_num_0 = np.ones(num_words)
    word_num_1 = np.ones(num_words)  # 初始化各个词在侮辱和正常言论中出现的次数
    num_0 = 2.0
    num_1 = 2.0
    for i in range(num_train_docs):
        if train_y[i] == 1:
            word_num_1 += train_X[i]
            num_1 += sum(train_X[i])
        else:
            word_num_0 += train_X[i]
            num_0 += sum(train_X[i])
    # 计算每个词在侮辱和正常言论中的出现的概率,为了防止结果下溢出
    p1_vec = np.log(word_num_1/num_1)
    p0_vec = np.log(word_num_0/num_0)
    return p0_vec, p1_vec, p_abusive

def classify(vec2classify, p0_vec, p1_vec, p_class1):
    """
    由于在贝叶斯公司中，分母是一样的都是p(w)，所以比较分子就行
    :param vec2classify: 待分类的向量
    :param p0_vec:
    :param p1_vec:
    :param p_class1:
    :return:
    """
    p0 = sum(vec2classify * p0_vec) + np.log(1-p_class1)
    p1 = sum(vec2classify * p1_vec) + np.log(p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    post_list,  class_list= load_dataset()
    vocab_list = create_vocab_list(post_list)
    train_mat=[]
    for post in post_list:
        train_mat.append(set_of_word2vec(vocab_list, post))
    p0v, p1v,pab = train_nb(train_mat, class_list)

