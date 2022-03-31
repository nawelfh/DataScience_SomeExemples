# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:54:04 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [2,3]].values

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting herarchical clustering to the data
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
Y_hc=hc.fit_predict(X)

#visualising the clusters
plt.scatter( X[Y_hc==0,0],  X[Y_hc==0,1],s=100,color='gray',label='Cluster1')
plt.scatter( X[Y_hc==1,0], X[Y_hc==1,1],s=100,color='blue',label='Cluster2')
plt.scatter( X[Y_hc==2,0], X[Y_hc==2,1],s=100,color='brown',label='Cluster3')
plt.scatter( X[Y_hc==3,0], X[Y_hc==3,1],s=100,color='black',label='Cluster4')
plt.scatter( X[Y_hc==4,0], X[Y_hc==4,1],s=100,color='magenta',label='Cluster5')
plt.scatter( X[Y_hc==5,0], X[Y_hc==5,1],s=100,color='red',label='Cluster6')
plt.scatter( X[Y_hc==6,0], X[Y_hc==6,1],s=100,color='green',label='Cluster7')
plt.scatter( X[Y_hc==7,0], X[Y_hc==7,1],s=100,color='orange',label='Cluster8')
plt.scatter( X[Y_hc==8,0], X[Y_hc==8,1],s=100,color='yellow',label='Cluster9')
plt.scatter( X[Y_hc==9,0], X[Y_hc==9,1],s=100,color='purple',label='Cluster10')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Annual Income(K$)')
plt.legend()
plt.show()