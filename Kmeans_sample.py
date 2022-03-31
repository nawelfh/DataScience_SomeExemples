# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:14:29 2020

@author: king info
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


Kmean=KMeans(n_clusters=2)
Kmean.fit(X_train)
Kmean.cluster_centers_

plt.scatter(X_train[:,0],X_train[:,1],s=50)
plt.scatter(0.55834773, 1.23388568,s=200,color='green',marker='s')
plt.scatter(-0.22074213,-0.48781527,s=200,color='red',marker='s')
plt.show()

Z=Kmean.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Z)



