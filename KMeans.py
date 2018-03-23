#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Create on: 2018/3/20 0020 16:09 
# Author: Lyu 
# Annotation: 优点：容易实现。缺点：可能收敛到局部最小值，大规模数据集上收敛较慢。

import numpy as np

def load_data_set(filename):
    data_mat = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            cur_line = line.strip().split('\t')
            flt_line = map(np.float, cur_line)
            data_mat.append(flt_line)
    return data_mat

def dist_clud(vecA, vecB):
    """欧式距离公式"""
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def rand_cent(dataset, k):
    """
    随机确定簇中心
    分别对每个特征随机取值
    """
    dataset = np.mat(dataset)
    n = np.shape(dataset)[1]
    cent_roids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minj = np.min(dataset[:, j])
        rangej = np.float(np.max(dataset[:, j]) - minj)
        cent_roids[:, j] = minj+rangej*np.random.rand(k, 1)
    return cent_roids

def kmeans(dataset, k, distmeas=dist_clud, create_cent=rand_cent):
    if not isinstance(dataset, np.matrix):
        dataset = np.mat(dataset)
    m = np.shape(dataset)[0]
    cluster_assment = np.mat(np.zeros((m, 2))) # 每个点的簇分配结果
    cent_roids = create_cent(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                distJI = distmeas(cent_roids[j, :], dataset[i, :]) # 计算与簇中心的距离
                if distJI < min_dist:
                    min_dist = distJI
                    min_index = j
            if cluster_assment[i,0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist**2
        print cent_roids

        for cent in range(k):
            pts_in_clust = dataset[np.nonzero(cluster_assment[:, 0].A == cent)[0]] # nonzero 返回非0值的下标pt
            cent_roids[cent, :] = np.mean(pts_in_clust, axis=0)
    return cent_roids, cluster_assment


if __name__ == '__main__':
    dataset = load_data_set("kmeans_testset.txt")
    # rand_cent(dataset, 3)
    kmeans(dataset, 3)