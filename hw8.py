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
from sklearn.svm import SVC

hw8_csv = pd.read_csv('data/hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype = np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

model = SVC(kernel = 'rbf', C = 1.0, gamma = 0.1)
model.fit(X0, y)

x_min, x_max = X0[:, 0].min() - 1, X0[:, 0].max() + 1
y_min, y_max = X0[:, 1].min() - 1, X0[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

fig = plt.figure(dpi=288)
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Distribution')
plt.axis('equal')
plt.legend()
plt.show()

fig = plt.figure(dpi=288)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Distribution')
plt.axis('equal')
plt.legend()
plt.show()
