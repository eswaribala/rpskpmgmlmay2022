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
#print(df.values)
#scaling techniques
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)

print(df.iloc[1:10,0])
print(df.iloc[1:10,1])
#stdscalar=preprocessing.StandardScaler()
#x_scaled = stdscalar.fit_transform(df)
#df = pd.DataFrame(x_scaled)
#print(df)
import matplotlib.pyplot as plt 
plt.figure()
plt.title('Angola Poulation')
plt.xlabel('Year')
plt.ylabel('Population')
plt.plot(df.iloc[1:1000,0], df.iloc[1:1000,1], 'b.') #color code is k or m or etc.,
plt.axis([0.1, 1, 0.01,0.1])
plt.grid(True)
plt.show()