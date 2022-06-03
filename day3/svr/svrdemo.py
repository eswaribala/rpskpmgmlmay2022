# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:56:16 2022

@author: Balasubramaniam
"""
import pandas as pd

df=pd.read_csv("Data.csv")
x=df.iloc[:,1:2].values
y=df.iloc[:,2:3].values
print(x)
print(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

y_pred = regressor.predict([[46]])
print(y_pred)
