# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:47:23 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.cluster import KMeans
Kmean=KMeans(n_clusters=2)
Kmean.fit(X_train)
Kmean.cluster_centers_
plt.scatter(X_train[:,0],X_train[:,1],s=50)
plt.scatter(0.55834773, 1.23388568,s=200,color='green',marker='s')
plt.scatter(-0.22074213,-0.48781527,s=200,color='red',marker='s')
plt.show()
a=Kmean.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, a)

from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
               np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, Kmean.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                       alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
      plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                          c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kmeans Training')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()