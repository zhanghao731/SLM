import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# plt.scatter(X[:50, 0], X[:50, 1], label = '0')
# plt.scatter(X[50:, 0], X[50:, 1], label = '1')
# plt.legend()
# plt.show()
# class SVM:
#     def __init__(self, max_iter = 100, kernel = 'linear'):
#         self.max_iter = max_iter
#         self._kernel = kernel
#
#     def init_args(self, features, labels):
#         self.m, self.n = features.shape
#         self.X = features
#         self.Y = labels
#         self.b = 0.0
#
#         # 将Ei保存在一个列表里
#         self.alpha = np.ones(self.m)
#         self.E = [self._E(i) for i in range(self.m)]
#         # 松弛变量
#         self.C = 1.0
#
#     def kernel(self, x1, x2):
#         if self.kernel == 'linear':
#             return sum([x1[k] * x2[k] for k in range(self.n)])
#         elif self.kernel == 'poly':
#             return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
#         return 0
#
#     def _g(self, i):
#         r = self.b
#         for j in range(self.m):
#             r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
#         return r
#
#     def _E(self, i):
#         return self._g(i) - self.Y[i]
#
#     def _KKT(self,i):
#         y_g=self._g(i)*self.Y[i]
#         if

from sklearn.svm import SVC

clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
