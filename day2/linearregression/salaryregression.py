# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:23:04 2022

@author: Balasubramaniam
"""
import matplotlib.pyplot as plt 

import pandas as pd

df=pd.read_csv("Salary_Data.csv")
x=df["YearsExperience"]
y=df["Salary"]
#scaling techniques
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)
print(df.iloc[:,0].values)
print(df.iloc[:,1].values)


plt.figure()
plt.title('Salary Data')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.plot(df.iloc[:,0], df.iloc[:,1], 'm.') #color code is k or m or etc.,
plt.axis([-0.1, 1.2, -0.1, 1.2])
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,0].values,df.iloc[:,1].values,test_size=0.3,random_state=0)

print("x training and test")
print(x_train)
print(x_test)
print("x training and test")
print(y_train)
print(y_test)

import numpy as np

x_train=np.array(x_train).reshape(-1,1)
print(x_train)
y_train=np.array(y_train).reshape(-1,1)

x_test=np.array(x_test).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)


#linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(x_train,y_train) 
print("Results.....")
print(model.coef_)
print(model.intercept_)

#prediction
y_pred=model.predict(x_test)
print("Y Prediction")
print(y_pred)

#error 

#MSE

import numpy as np
#Mean Sqaured Error
MSE=np.mean((y_test-y_pred)**2)
print("Mean Sqaured Error %r" %(MSE))
#Sum of Sqaured Error
SSE=np.sum((y_test-y_pred)**2)
print("SUM of Sqaured Error %r" %(SSE))

print("Accuracy level",(1-MSE))



# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, model.predict(x_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''
import statsmodels.api as sm

est = sm.OLS(y_train, x_train) #ordinary least square method 
est2 = est.fit()
print("Summary.....")
print(est2.summary())
'''


