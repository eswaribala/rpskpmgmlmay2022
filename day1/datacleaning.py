# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:02:46 2022

@author: Balasubramaniam
"""
import pandas as pd

df=pd.read_csv("Data.csv")
#print(df)
x = df.iloc[:, :-1].values
#print(x)
y = df.iloc[:,3].values #dataset.iloc[:,3:]


#print(x)
# Taking care of missing data
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
print(x[:,2])
print(x[:,1])
