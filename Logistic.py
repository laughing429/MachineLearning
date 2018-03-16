#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018/3/15 0015 11:47 
# Author: Lyu

# 优点：计算代价不高，易于理解和实现
# 缺点：容易欠拟合，分类精度可能不高
# 适用数据类型：数值型和标称型数据

# sigmoid = 1/(1+e^-z)
# 梯度上升法：要找到某函数的最大值，最好的方法就是沿着函数的题都方向探寻

import numpy as np
from matplotlib import pyplot as plt

def load_dataset():
    data_mat = []
    label_mat = []
    with open("logistic_dataset.txt", 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            # todo 为什么这里每条数据前面要加1.0
            data_mat.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
            label_mat.append(np.int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


def grad_ascent(datamat_in, class_labels):
    data_matrix = np.mat(datamat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500 # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix*weights)
        error = label_mat - h
        weights += alpha*data_matrix.transpose()*error # 梯度下降的话就是加号改成减号
    return weights


def stoc_grad_ascent(datamat, class_labels, num_iter=150):
    """随机梯度上升"""
    m,n = np.shape(datamat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(datamat[randIndex]*weights))
            error = class_labels[randIndex] - h
            weights = weights + alpha*error*datamat[randIndex]
            del data_index[randIndex]
    return weights


def plot_best_fit(weight):
    # weight = weight.getA() # 由矩阵转换成向量
    datamat, labelmat = load_dataset()
    data_arr = np.array(datamat)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xcord1.append(data_arr[i,1])
            ycord1.append(data_arr[i,2])
        else:
            xcord2.append(data_arr[i,1])
            ycord2.append(data_arr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 画最佳拟合线 w^T*X = 0
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1]*x)/weight[2]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    datamat, labelmat = load_dataset()
    weight = grad_ascent(datamat, labelmat)
    plot_best_fit(weight)