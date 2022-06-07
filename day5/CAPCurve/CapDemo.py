# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 06:59:50 2022

@author: Balasubramaniam
"""
# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the dataset and define the input and output features
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Define the Random Forest model, train the model and make prediction on test data
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)


#Define the Logsistic Regression model, train the model and make prediction on test data
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)


#Define the KNN model, train the model and make prediction on test data
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)


#Define the Naive Bayes model, train the model and make prediction on test data
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)


#Visualize the CAP Curve Analysis including all 4 classification models
total = len(y_test) 
one_count = np.sum(y_test) 
zero_count = total - one_count 
lm_rf = [y for _, y in sorted(zip(y_pred_rf, y_test), reverse = True)]
lm_lr = [y for _, y in sorted(zip(y_pred_lr, y_test), reverse = True)] 
lm_knn = [y for _, y in sorted(zip(y_pred_knn, y_test), reverse = True)] 
lm_nb = [y for _, y in sorted(zip(y_pred_nb, y_test), reverse = True)] 
x = np.arange(0, total + 1) 
y_rf = np.append([0], np.cumsum(lm_rf)) 
y_lr = np.append([0], np.cumsum(lm_lr)) 
y_knn = np.append([0], np.cumsum(lm_knn)) 
y_nb = np.append([0], np.cumsum(lm_nb)) 
plt.figure(figsize = (10, 6)) 
plt.plot([0, total], [0, one_count], c = 'b', linestyle = '--', label = 'Random Model')
plt.plot([0, one_count, total], [0, one_count, one_count], c = 'grey', linewidth = 2, label = 'Perfect Model')
plt.title('CAP Curve of Classifiers')
plt.plot(x, y_rf, c = 'b', label = 'RF classifier', linewidth = 2)
plt.plot(x, y_lr, c = 'r', label = 'LR classifier', linewidth = 2)
plt.plot(x, y_knn, c = 'y', label = 'KNN classifier', linewidth = 2)
plt.plot(x, y_nb, c = 'm', label = 'NB classifier', linewidth = 2)
plt.legend()