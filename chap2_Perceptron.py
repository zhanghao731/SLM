import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
# print(iris)
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# print(df.columns)
# print(df.label.value_counts())
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label = '0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label = '1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
# print(X, y)
y = np.array([1 if i == 1 else -1 for i in y])


# Perceptron
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype = np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # SGD
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True

        return "Perceptron Model"

    def score(self):
        pass


# perceptron = Model()
# perceptron.fit(X, y)
# x_points = np.linspace(4, 7, 10)
# y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
# plt.plot(x_points, y_)
#
# plt.plot(data[:50, 0], data[:50, 1], 'bo', color = 'blue', label = '0')
# plt.plot(data[50:100, 0], data[50:100, 1], "bo", color = 'orange', label = '1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()


# using sklearn
clf = Perceptron(fit_intercept = True, max_iter = 1000, shuffle = True, tol = None)
clf.fit(X, y)

print(clf.coef_)  # weights assigned to the features
print(clf.intercept_)  # Constants in decision function.
print(type(clf.intercept_))
# 画布大小
plt.figure(figsize = (10, 10))
# 中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c = 'b', label = 'Iris-setosa')
plt.scatter(data[50:100, 0], data[50:100, 1], c = 'orange', label = 'Iris-versicolor')

# 感知机的线
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)

plt.legend()
plt.grid(False)
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()
plt.show()
