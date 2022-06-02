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
y_pred=model.predict(y_test)
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





