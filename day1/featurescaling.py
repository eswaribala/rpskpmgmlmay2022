# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:48:22 2022

@author: Balasubramaniam
"""
import pandas as pd

df=pd.read_csv("population.csv")
x=df.iloc[:,2]
y=df.iloc[:,3]
#print(x)
#print(y)
df = pd.DataFrame( {"year":df["Year"], "population":df["Value"]})
print(df.values)
#scaling techniques
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)
print(df)

stdscalar=preprocessing.StandardScaler()
x_scaled = stdscalar.fit_transform(df)
df = pd.DataFrame(x_scaled)
print(df)
