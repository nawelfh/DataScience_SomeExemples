# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:21:58 2020

@author: king info
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("50_Startups.csv")

X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
labelEncoder.fit(X.iloc[:,[3]])
X.iloc[:,[3]]=labelEncoder.transform(X.iloc[:,[3]])

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)

from sklearn import metrics
MAE=metrics.mean_absolute_error(y_pred,Y_test)
MSE=metrics.mean_squared_error(y_pred,Y_test)
RMSE=metrics.mean_squared_error(y_pred,Y_test)**0.5

regressor.predict(np.array([[0,0,130000,140000,300000]]))

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()