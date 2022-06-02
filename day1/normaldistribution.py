# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:22:49 2022

@author: Balasubramaniam
"""
from scipy.stats import uniform
import seaborn as sns
import matplotlib.pyplot as plt
# random numbers from uniform distribution
# Generate 10 numbers from 0 to 10
n = 100 # Generate 100000 numbers
a = 0
b = 10
data_uniform = uniform.rvs(size=n, loc = a, scale=b)   
ax = sns.distplot(data_uniform,
                  bins=10,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform ', ylabel='Frequency')

plt.show()