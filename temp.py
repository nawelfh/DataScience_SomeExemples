# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1]
X1=dataset.iloc[:,:-1]
X2=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X.iloc[:,[1,2]])
X.iloc[:,1:3]=imputer.transform(X.iloc[:,1:3])

imputer=Imputer(missing_values='NaN',strategy='median',axis=0)
imputer.fit(X1.iloc[:,[1,2]])
X1.iloc[:,1:3]=imputer.transform(X1.iloc[:,1:3])

imputer=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer.fit(X2.iloc[:,[1,2]])
X2.iloc[:,1:3]=imputer.transform(X2.iloc[:,1:3])

plt.hist(X.iloc[:,1])
plt.hist(X1.iloc[:,1])
plt.hist(X2.iloc[:,1])

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
labelEncoder.fit(Y)
Y=labelEncoder.transform(Y)

labelEncoder=LabelEncoder()
labelEncoder.fit(X.iloc[:,0])
X.iloc[:,0]=labelEncoder.transform(X.iloc[:,0])

from sklearn.preprocessing import OneHotEncoder

oneHotEncoder=OneHotEncoder(categorical_features=[0])
oneHotEncoder.fit(X)
X=oneHotEncoder.transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

