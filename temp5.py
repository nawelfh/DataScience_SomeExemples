import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
dataset = pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,-1].values
regressor=LinearRegression()
regressor.fit(X,Y)
#plt.scatter(X,Y,color='red')
#plt.plot(X,regressor.predict(X),color='blue')
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
regressor.fit(X_poly,Y)
#plt.scatter(X,Y,color='red')
#plt.plot(X,regressor.predict(X_poly),color='blue')
#plt.title('Salary vs Level (Polynomial Regression degree 4)')
#plt.xlabel('Level')
#plt.ylabel('Salary')
#plt.show()
Constant=regressor.intercept_
Coefficients=regressor.coef_
Y_pred=regressor.predict(X_poly)
RMSE=np.sqrt(mean_squared_error(Y_pred,Y))
#rmses=[]
#for i in range(2,18):
#    poly_reg=PolynomialFeatures(degree=i)
#    X_poly=poly_reg.fit_transform(X)
#    regressor.fit(X_poly,Y)
#    Constant=regressor.intercept_
#    Coefficients=regressor.coef_
#    Y_pred=regressor.predict(X_poly)
#    RMSE=np.sqrt(mean_squared_error(Y_pred,Y))
#    rmses.append(RMSE)
#minimum= min(rmses) 
#print("Minimum is "+str(minimum)+ " with index : "+str(rmses.index(minimum)+1))  
MAE=metrics.mean_absolute_error(Y_pred,Y)
MSE=metrics.mean_squared_error(Y_pred,Y)
RMSE=MSE**0.5
