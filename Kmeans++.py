# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:28:44 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [2,3]].values

from sklearn.cluster import KMeans
#wcss = []
#for i in range (1,11):
#    Kmeans = KMeans(n_clusters=i,random_state=42)
#    Kmeans.fit(X)
#    wcss.append(Kmeans.inertia_)
#plt.plot(range(1,11),wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

kmeans=KMeans(n_clusters=4,random_state=42)
kmeans.fit(X)
cluster_centers=kmeans.cluster_centers_
y_set=kmeans.labels_

from matplotlib.colors import ListedColormap
for i, j in enumerate(np.unique(y_set)):
      plt.scatter(X[y_set == j, 0], X[y_set == j, 1],
                          c = ListedColormap(('red', 'green','yellow','purple','orange','brown','blue','pink','gray','cyan','black'))(i), label = j)
plt.scatter( cluster_centers[0,0], cluster_centers[0,1],s=200,color='green',marker='s')
plt.scatter(cluster_centers[1,0], cluster_centers[1,1],s=200,color='red',marker='s')
plt.scatter(cluster_centers[2,0], cluster_centers[2,1],s=200,color='blue',marker='s')
plt.scatter(cluster_centers[3,0], cluster_centers[3,1],s=200,color='yellow',marker='s')

X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
               np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kmeans.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                       alpha = 0.4, cmap = ListedColormap(('red', 'green','orange','purple','black','brown','magenta','pink','gray','cyan','yellow')))
plt.title('Kmeans (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()