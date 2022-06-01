# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:39:30 2022

@author: Balasubramaniam
"""

import pandas as pd

df=pd.read_csv("Data.csv")
print(df)
print ("Dataframe", df)
print ("Shape", df.shape)
print ("Length", len(df))
print ("Column Headers", df.columns)
print ("Data types", df.dtypes)
#print("Index", df.index)
#print ("Values", df.values)
#print(df.head(2))
#print(df.tail(2))

#pritn rows
rows=df.iloc[:,:-1].values
column=df.iloc[:,3].values
print(rows)
print(column)



#print("Average Glucose", df["Glucose"].mean())



'''

import requests
import json

data=requests.get("https://jsonplaceholder.typicode.com/users")
jsondata=json.loads(data.text)
print(jsondata)
df=pd.DataFrame.from_dict(jsondata)

print(df)
print ("Dataframe", df)
print ("Shape", df.shape)
print ("Length", len(df))
print ("Column Headers", df.columns)
print ("Data types", df.dtypes)
print("Index", df.index)
print ("Values", df.values)
print(df.head(2))
print(df.tail(2))

'''