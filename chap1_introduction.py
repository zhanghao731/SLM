import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)


# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差
def resduals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


# 十个点，等差数列
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

regularization = 0.0001


def fitting(M = 0):
    """M表示多项式次数"""
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(resduals_func, p_init, (x, y))
    print('Fitting Parameters:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label = "real")
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label = "fitted curve")
    plt.plot(x, y, 'bo', label = 'noise')
    plt.legend()
    plt.show()
    return p_lsq


def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
    return ret


p_lsq_9 = fitting(M = 9)
p_init = np.random.rand(9 + 1)
p_lsq_regularization = leastsq(residuals_func_regularization, p_init, (x, y))
plt.plot(x_points, real_func(x_points), label = 'real')
plt.plot(x_points, fit_func(p_lsq_9[0],x_points),label='fitted curve')
plt.plot(x_points,fit_func(p_lsq_regularization[0],x_points),label='regularization')
plt.plot(x,y,'bo',label='noise')
plt.legend()
plt.show()


