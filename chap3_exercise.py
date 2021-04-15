import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = np.array(
    [[5, 12, 1], [6, 21, 0], [14, 5, 0], [16, 10, 0], [13, 19, 0], [13, 32, 1], [17, 27, 1], [18, 24, 1],
     [20, 20, 0], [23, 14, 1], [23, 25, 1], [23, 31, 1], [26, 8, 0], [30, 17, 1], [30, 26, 1], [34, 8, 0], [34, 19, 1],
     [37, 28, 1]])
X_train = data[:, 0:2]
y_train = data[:, 2]
models = (KNeighborsClassifier(n_neighbors = 1, n_jobs = -1), KNeighborsClassifier(n_neighbors = 2, n_jobs = -1))
models = (clf.fit(X_train, y_train) for clf in models)
titles = ('K Neighbors with k=1', 'K Neighbors with k=2')
fig = plt.figure(figsize = (15, 5))
plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
X0, X1 = X_train[:, 0], X_train[:, 1]
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
# print(xx.shape, yy.shape)
# print(xx.ravel().shape, yy.ravel().shape)
# print(np.c_[xx.ravel(), yy.ravel()].shape)
# for clf, title, ax in zip(models, titles, fig.subplots(1, 2).flatten()):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(Z))])
#     ax.contourf(xx, yy, Z, cmap = cmap, alpha = 0.5)
#     ax.scatter(X0, X1, c = y_train, s = 50, edgecolors = 'k', cmap = cmap, alpha = 0.5)
#     ax.set_title(title)
# plt.show()

# import numpy as ap
# from sklearn.neighbors import KDTree
#
# train_data = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
# # print(train_data)
# tree = KDTree(train_data, leaf_size = 2)
# dist, ind = tree.query(np.array([(3, 3.5)]), k = 1)
# x1 = train_data[ind[0]][0][0]
# x2 = train_data[ind[0]][0][1]
# print("x的近邻点是({0},{1})".format(x1, x2))

from collections import namedtuple
import numpy as np


# 节点类
class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return str(tuple(self))


class KdTree:
    def __init__(self, k = 1):
        self.k = k
        self.kdtree = None

    # 构建kd tree
    def _fit(self, X, depth = 0):
        try:
            k = self.k
        except IndexError as e:
            return None
        axis = depth % k
        X = X[X[:, axis].argsort()]
        median = X.shape[0] // 2
        try:
            X[median]
        except IndexError:
            return None
        return Node(location = X[median], left_child = self._fit(X[:median], depth + 1),
                    right_child = self._fit(X[median:], depth + 1))

    def _search(self, point, tree = None, depth = 0, best = None):
        if tree is None:
            return best
        k = self.k
        if point[0][depth % k] < tree.location[depth % k]:
            next_branch = tree.left_child
        else:
            next_branch = tree.right_child
        if not next_branch is None:
            best = next_branch.location
        return self._search(point, tree = next_branch, depth = depth + 1, best = best)
