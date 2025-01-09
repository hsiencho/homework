# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

w = np.linalg.inv(Sw) @ (m2 - m1).T

w /= np.linalg.norm(w)

X1_proj = X1 @ w
X2_proj = X2 @ w

plt.figure(dpi=288)
plt.scatter(X1[:, 0], X1[:, 1], color='red', alpha=0.5, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='green', alpha=0.5, label='Class 2')

proj_line = np.linspace(-2, 5, 100).reshape(-1, 1) @ w.T
plt.plot(proj_line[:, 0], proj_line[:, 1], 'b-', label='LDA Projection Line')

plt.title('LDA Projection in 2D Space')
plt.legend()
plt.axis('equal')
plt.show()