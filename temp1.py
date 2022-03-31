import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
dataset = pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
plt.scatter(X,Y,color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
a=regressor.predict([[15]])
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.plot(X_test,regressor.predict(X_test),color='green')
plt.show()

MAE=metrics.mean_absolute_error(y_pred,Y_test)
MSE=metrics.mean_squared_error(y_pred,Y_test)
RMSE=metrics.mean_squared_error(y_pred,Y_test)**0.5

x=dataset.iloc[:,0].values
#r=(np.mean(x*Y)-np.mean(x)*np.mean(Y))/(np.std(x)*np.std(Y))
sigma=np.corrcoef(x,Y)

b1=regressor.coef_
b0=regressor.intercept_

score=r2_score(Y_test,y_pred)
sc=np.sqrt(score)