#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018/3/19 0019 11:43 
# Author: Lyu

# 我们可以将不同的分类器组合起来，而这种组合结果则被称为集成方法或者元算法。使用集成方法时，会有多种形式，可以是不同算法的集成，也可以是同一算法在不同设置下的集成，还可以是数据集不同部分分配给不同分类器之后的集成。
# 某个分类器的误差率等于该分类器的被错分样本的权值之和

import numpy as np

def load_simp_data():
    datmat = np.matrix([[1.0, 2.1],
                        [2.0, 1.1],
                        [1.3, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datmat, class_labels

def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """构建符号函数"""
    m,n = np.shape(data_matrix)
    ret_array = np.ones((m, 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = 1.0
    return ret_array

def build_stump(data_arr, class_labels, D):
    data_mat = np.matrix(data_arr)
    label_mat = np.matrix(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_clas_est = np.mat(np.zeros(m, 1))
    min_error = np.inf
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max-range_min)/num_steps
        for j in range(-1, np.int(num_steps)+1):
            for inequal in ['lt', 'gt']:
                thresh_val = range_min+np.float(j)*step_size
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T*err_arr
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_clas_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_clas_est
