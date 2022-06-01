# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:11:06 2022

@author: Balasubramaniam
"""
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("diabetes.csv")
x=df["BMI"]
y=df["Glucose"]
y1=df["BloodPressure"]
plt.scatter(x, y, color='g')
plt.scatter(x, y1, color='m')
plt.show()

plt.bar(x, y, color='g', align='center')

plt.show()

import seaborn as sb
sb.barplot(x = "BMI", y = "BloodPressure",  data = df)
plt.show()