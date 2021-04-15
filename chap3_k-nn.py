import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


# L_p distance
def L(x, y, p = 2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    return 0


# x1 = [1, 1]
# x2 = [5, 1]
# x3 = [4, 4]
#
# for i in range(1, 5):
#     r = {'1-{}'.format(c): L(x1, c, p = i) for c in [x2, x3]}
#     print(min(zip(r.values(), r.keys())))

# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


class KNN:
    def __init__(self, X_train, y_train, n_neighbors = 3, p = 2):
        """
        parameter:n_neighbors 临近点个数
        parameter:p 距离度量
        """

        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord = self.p)
            knn_list.append((dist, self.y_train[i]))
        for i in range(self.n, len(self.X_train)):
            # 找出n个点中与目标点距离最远的点，当有更近的点时替换掉
            max_index = knn_list.index(max(knn_list, key = lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord = self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key = lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if y == label:
                right_count += 1
        return right_count / len(X_test)


# clf = KNN(X_train, y_train)
# print(clf.score(X_test, y_test))
#
# test_point = [6.0, 3.0]
# print('Test Point: {}'.format(clf.predict(test_point)))
# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
# plt.plot(test_point[0], test_point[1], 'bo', label = 'test_point')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

# scikit-learn实例
# from sklearn.neighbors import KNeighborsClassifier
#
# clf_sk = KNeighborsClassifier()
# clf_sk.fit(X_train, y_train)
# print(clf_sk.score(X_test, y_test))

# kd_tree
class KdNode:  # kd树节点的数据结构
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点（k维空间中的一个样本点）
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 分割后左子空间构成的kd树
        self.right = right  # 右子空间构成的kd树


class KdTree:
    def __init__(self, data):
        # 数据维度
        k = len(data[0])

        def CreateNode(split, data_set):  # 按第split维划分数据集，创建KdNode
            if not data_set:
                return None
            # 排序
            data_set.sort(key = lambda x: x[split])
            split_pos = len(data_set) // 2
            mid = data_set[split_pos]
            split_next = (split + 1) % k
            # 递归创建kd树
            return KdNode(mid, split, CreateNode(split_next, data_set[:split_pos]),
                          CreateNode(split_next, data_set[split_pos + 1:]))

        self.root = CreateNode(0, data)


# 前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


# 对构建好的kd树进行搜索，寻找与目标点最近的样本点
from math import sqrt
from collections import namedtuple

# 定义一个namedtuple，分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple('Result_tuple', 'nearest_point nearest_dist nodes_visited')


def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):  # 搜索某个区域的最近点
        if kd_node is None:
            return result(nearest_point = [0] * k, nearest_dist = float('inf'), nodes_visited = 0)
        nodes_visited = 1
        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的轴
        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            further_node = kd_node.right

        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)
        nearest = temp1.nearest_point  # 当前最近点
        dist = temp1.nearest_dist  # 更新最近距离
        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist
        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist:  # 不与超平面相交
            return result(nearest, dist, nodes_visited)
        # 目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
        if temp_dist < dist:  # 如果更近
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超体球半径
        # 相交时需要检查另一个子节点对应区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)
        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float('inf'))


data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
# preorder(kd.root)

import time
from random import random


# 产生一个k维随机向量，分量在0-1之间
def random_point(k):
    return [random() for _ in range(k)]


def random_points(k, n):
    return [random_point(k) for _ in range(n)]


ret = find_nearest(kd, [3, 4.5])
print(ret)

N = 400000
t0 = time.perf_counter()
kd2 = KdTree(random_points(3, N))
ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])
t1 = time.perf_counter()
print('time: ', t1 - t0, 's')
print(ret2)
