import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
import pprint


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = ['年龄', '有工作', '有自己的孩子', '信贷情况', '类别']
    return datasets, labels


datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns = labels)


# print(train_data)


# print(info_gain_train(np.array(datasets)))

# 定义节点类，二叉树
class Node:
    def __init__(self, root = True, label = None, feature_name = None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}

    def __repr__(self):
        result = {'label:': self.label, 'feature:': self.feature, 'tree:': self.tree}
        return '{}'.format(result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon
        self._tree = {}

    @staticmethod
    # 计算熵
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent

    # 计算经验条件熵
    def cond_ent(self, datasets, axis = 0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    @staticmethod
    # 信息增益
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        # 特征数量
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis = c))
            best_feature.append((c, c_info_gain))
            # print('特征（{}）- info_gain - {:.3f}'.format(labels[c], c_info_gain))
        best_ = max(best_feature, key = lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式),特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 1.只有一类标记
        if len(y_train.value_counts()) == 1:
            return Node(root = True, label = y_train.iloc[0])
        # 2.特征为空
        if len(features) == 0:
            return Node(root = True, label = y_train.value_counts().sort_values(ascending = False).index[0])
        # 3.计算最大信息增益及对应的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]
        # 4.与阈值比较
        if max_info_gain < self.epsilon:
            return Node(root = True, label = y_train.value_counts().sort_values(ascending = False).index[0])
        # 5.构建子集
        node_tree = Node(root = False, feature_name = max_feature_name, feature = max_feature)
        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis = 1)
            # 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)


# dt = DTree()
# tree = dt.fit(train_data)
# print(tree)
# print(dt.predict(['老年', '否', '否', '一般']))

# scikit-learn实例
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz  # 决策树可视化
import graphviz

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train, )
clf.score(X_test, y_test)
tree_pic = export_graphviz(clf, out_file = 'mytree.pdf')
with open('mytree.pdf') as f:
    dot_graph = f.read()
print(graphviz.Source(dot_graph))
