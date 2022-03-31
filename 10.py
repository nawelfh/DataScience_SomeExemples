# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:12:49 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[2,3]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow Methode')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#cluster 4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X)
from matplotlib.colors import ListedColormap
X_set, y_set = X,kmeans.labels_
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kmeans.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','yellow','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('pink', 'green','white','blue'))(i), label = j)
plt.title('KMeans Train Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.scatter(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],s=200,color='black',marker='s')
plt.scatter(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],s=200,color='black',marker='s')
plt.scatter(kmeans.cluster_centers_[2][0],kmeans.cluster_centers_[2][1],s=200,color='black',marker='s')
plt.scatter(kmeans.cluster_centers_[3][0],kmeans.cluster_centers_[3][1],s=200,color='black',marker='s')
plt.show()