# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:34:20 2022

@author: Balasubramaniam
"""
import pandas as pd
df=pd.read_csv("Data.csv")
x=df.iloc[:,0:3]
#print(x)

y=df.iloc[:,3]
#print(y)

#split the data

#random state controls the way shuffling happens when every time the code runs
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

print(x_train)
print(x_test)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

print(x_train)
print(x_test)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

print(x_train)
print(x_test)