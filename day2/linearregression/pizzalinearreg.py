# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 08:38:54 2022

@author: Balasubramaniam
"""
import matplotlib.pyplot as plt
x = [[6], [8], [10], [14], [18]] #pizza diameter
print(x[0])
y = [[7], [9], [13], [15], [18]] #price in dollars
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(x, y, 'm.') #color code is k or m or etc.,
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

#simple linear regression
from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(x,y)
print ('A 9" pizza should cost: $%.2f' % linearRegression.predict([[9]]))

import numpy as np
MSE=np.mean((10-linearRegression.predict([[9]]))**2)
print("Mean Sqaured Error %r" %(MSE))
