# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:55:47 2022

@author: Balasubramaniam
"""
import  pandas as pd

df=pd.read_csv("https://data.nasdaq.com/api/v3/datasets/OPEC/ORB.csv?collapse=monthly")
print(df)




#import requests
#import json

#data=requests.get("https://jsonplaceholder.typicode.com/users")

#testdata=json.load(data)
