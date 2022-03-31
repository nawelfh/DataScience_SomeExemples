# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:08:33 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

X= -2*np.random.rand(100,2)
X1= 1+2*np.random.rand(50,2)

X[50:100,:]=X1


Kmean=KMeans(n_clusters=2)
Kmean.fit(X)
Kmean.cluster_centers_

plt.scatter(X[:,0],X[:,1],s=50)
plt.scatter(-0.95809327,-1.10070769,s=200,color='green',marker='s')
plt.scatter(1.98757962,2.06215874,s=200,color='red',marker='s')
plt.show()


Kmean.labels_
