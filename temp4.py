# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:33:51 2020

@author: king info
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm
dataset = pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1].values
labelEncoder=LabelEncoder()
X.iloc[:,-1]=labelEncoder.fit_transform(X.iloc[:,-1])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()
X=X[:,1:]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
regressor= LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
MAE=metrics.mean_absolute_error(y_pred,Y_test)
MSE=metrics.mean_squared_error(y_pred,Y_test)
RMSE=np.sqrt(MSE)
pred=regressor.predict([[0,1,130000,140000,300000]])
#S=np.array([1,2,1,0,1,2,1,0,1,2,1,0,1,2,1,2,1,2,1,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,0,1,2,1,2,1,2,1,2,1,2,1,2,1,0])
#HT=np.corrcoef(S,Y)
#U=np.c_[X,S]
#regressor1=LinearRegression()
#regressor1.fit(U,Y)
#Y_pred=regressor1.predict(U)
#r2=r2_score(Y,Y_pred)
#adjr2=1-(1-r2)*((50-1)/ (50-6-1))
#MAE=metrics.mean_absolute_error(Y_pred,Y_test)
#MSE=metrics.mean_squared_error(Y_pred,Y_test)
#RMSE=np.sqrt(MSE)
Constant=regressor.intercept_
Coefficients=regressor.coef_
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
summary=regressor_OLS.summary()
#for i in range (1,6):
#    X_opt=X[:,[0,i]]
#    regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#    summary=regressor_OLS.summary()
#    print("Summary with "+str(i))
#    print(summary)
#    print("----------------------")

#for i in range(1,6):
#    if i != 3:
#        X_opt=X[:,[0,3,i]]
#        regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#        summary=regressor_OLS.summary()
#        print("Summary with "+str(i)+" and 3")
#        print(summary)
#        print("----------------------")

#for i in range(1,6):
#    if i != 3 and i!= 5:
#        X_opt=X[:,[0,3,5,i]]
#        regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#        summary=regressor_OLS.summary()
#        print("Summary with "+str(i)+" and 3 and 5")
#        print(summary)
#        print("----------------------")

#for i in range(1,6):
#    if i != 3 and i!= 5 and i!=4:
#        X_opt=X[:,[0,3,5,4,i]]
#        regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#        summary=regressor_OLS.summary()
#        print("Summary with "+str(i)+" and 3 and 5 and 4")
#        print(summary)
#        print("----------------------")
        
#for i in range(1,6):
#    if i != 3 and i!= 5 and i!=4 and i!=1:
#        X_opt=X[:,[0,3,5,4,i]]
#        regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#        summary=regressor_OLS.summary()
#        print("Summary with "+str(i)+" and 3 and 5 and 4 and 1")
#        print(summary)
#        print("----------------------")


#X_opt=X[:,[0,3,5,4,1,2]]
#regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#summary=regressor_OLS.summary()
#print("Summary with all")
#print(regressor_OLS.f_pvalue )
#print("----------------------")
#def myFunction(X,variables):
#    currentPvalue=1
#    index=0
#    summary=None
#    for i in range(1,6):
#        array=[0]
#        for var in variables:
#            array.append(var)
#        if i not in variables:
#            array.append(i)
#            X_opt=X[:,array]
#            regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#            pvalue=regressor_OLS.f_pvalue
#            if pvalue < 0.05 and pvalue  < currentPvalue :
#                currentPvalue=pvalue
#                index=i
#                summary=regressor_OLS.summary()
#    return [index,summary]

#currentPvalue=1
#index=0
#summary=None
#for i in range(1,6):
#    X_opt=X[:,[0,i]]
#    regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
#    pvalue=regressor_OLS.f_pvalue
#    if pvalue < 0.05 and pvalue  < currentPvalue :
#        currentPvalue=pvalue
#        index=i
#        summary=regressor_OLS.summary()
#print(index)
#print(summary)
#variables=[]
#res=None
#for i in range(1,6):
#    res=myFunction(X,variables)
#    variables.append(res[0])
#print("--------------")
#print(variables)
#print("--------------")
#print(res[1])   


def mySecondFunction(X):
    variables=[]
    toExit=False
    for i in range(1,6):
        currentPvalue=1
        index=0
        summary=None
        for i in range(1,6):
            oldValue=currentPvalue
            array=[0]
            for var in variables:
                array.append(var)
            if i not in variables:
                array.append(i)
                X_opt=X[:,array]
                regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
                pvalue=regressor_OLS.f_pvalue
                if pvalue < 0.05 and pvalue  < currentPvalue :
                    currentPvalue=pvalue
                    index=i
                    summary=regressor_OLS.summary()
            if oldValue == currentPvalue:
                toExit=True
        if toExit:
            return variables[:-1]
        variables.append(index)
    return variables

res=mySecondFunction(X)
print(res)