# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:01:51 2017

@author: BALASUBRAMANIAM
"""

# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
# load the iris datasets
dataset = datasets.load_iris()
#print(dataset) # dispalys data and target
print(dataset.data)
print("target")
print(dataset.target_names)
print(dataset.feature_names)
print(dataset.DESCR)

X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                    dataset.target, 
                                                    test_size=0.2, random_state=0)

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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, predicted)
print("Confusion Matrix")
print(cm)


print ("Error Rate")


# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# Create DOT data
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=dataset.feature_names,  
                                class_names=dataset.target_names)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("iris2022.pdf")
