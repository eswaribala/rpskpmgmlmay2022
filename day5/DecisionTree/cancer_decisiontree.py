# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:01:51 2017

@author: BALASUBRAMANIAM
"""

# Decision Tree Classifier
#from sklearn import datasets
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import numpy as np

dataset = pd.read_csv('F:/citi_ml_jun2018/day4/KNN/breast-cancer-wisconsin.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values
print(X[:,5])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
#imputer = imputer.fit(X[:, 1:3])
#print(imputer)
X[:,5:6] = imputer.fit_transform(X[:,5:6])
print(X[:,5:6])
print(X)
print(y)
print(X.shape)
print(y.shape)

features=["ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",	"Mitoses"
]
targets=["Type2","Type4"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# fit a CART model to the data

model = DecisionTreeClassifier(criterion='entropy',
                       max_depth=4, random_state=0)
model.fit(X_train, y_train)
#print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
print(expected)
print(predicted)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Create DOT data
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=features,  
                                class_names=targets)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("diabetics2020.pdf")
