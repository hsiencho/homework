# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

# If this script is not run under spyder IDE, comment the following two lines.
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def scatter_pts_2d(x, y):
    # set plotting limits
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin,xmax,ymin,ymax

def analytic_gradient(w, x, y):
    w0, w1, w2, w3 = w
    e = y - (w0 + w1 * np.sin(w2 * x + w3))
    grad_w0 = -2 * np.sum(e)
    grad_w1 = -2 * np.sum(e * np.sin(w2 * x + w3))
    grad_w2 = -2 * np.sum(e * w1 * x * np.cos(w2 * x + w3))
    grad_w3 = -2 * np.sum(e * w1 * np.cos(w2 * x + w3))
    return np.array([grad_w0, grad_w1, grad_w2, grad_w3])

def numeric_gradient(w, x, y, epsilon=1e-5):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_forward = w.copy()
        w_backward = w.copy()
        w_forward[i] += epsilon
        w_backward[i] -= epsilon
        J_forward = np.sum((y - (w_forward[0] + w_forward[1] * np.sin(w_forward[2] * x + w_forward[3])))**2)
        J_backward = np.sum((y - (w_backward[0] + w_backward[1] * np.sin(w_backward[2] * x + w_backward[3])))**2)
        grad[i] = (J_forward - J_backward) / (2 * epsilon)
    return grad

dataset = pd.read_csv('data/hw7.csv').to_numpy(dtype = np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

alpha = 0.05
max_iters = 500
# cost function
#     J(w0, w1, w2, w3) = sum(y[i] - w0 - w1 * sin(w2 * x[i] + w3))^2
w_analytic = w.copy()
for _ in range(1, max_iters):
    grad = analytic_gradient(w_analytic, x, y)
    w_analytic -= alpha * grad


xmin,xmax,ymin,ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
w_numeric = w.copy()
for _ in range(1, max_iters):
    pass
    grad = numeric_gradient(w_numeric, x, y)  # 數值梯度
    w_numeric -= alpha * grad  # 更新參數
    
xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w_analytic[0] + w_analytic[1] * np.sin(w_analytic[2] * xt + w_analytic[3])
yt2 = w_numeric[0] + w_numeric[1] * np.sin(w_numeric[2] * xt + w_numeric[3])

# plot x vs y; xt vs yt1; xt vs yt2 
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
