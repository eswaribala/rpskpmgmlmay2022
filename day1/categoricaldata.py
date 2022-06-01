# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:41:31 2022

@author: Balasubramaniam
"""
import pandas as pd
df=pd.read_csv("Data.csv")
x=df.iloc[:,0:1].values
print(x)
y=df.iloc[:,3:].values
print(y)

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
#country encoding
x[:,0]=labelEncoder.fit_transform(x[:,0])
print(x)
#purchase encoding
labelEncoder=LabelEncoder()
#purchased encoding
y[:,0]=labelEncoder.fit_transform(y[:,0])
print(y)

'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_)
le.transform(["tokyo", "tokyo", "paris"])
print(list(le.inverse_transform([2, 2, 1])))
'''