# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:23:02 2022

@author: Yen-Chen Lai
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from math import sqrt,exp,pi

filename2 = './loss_data.csv'
data2 = pd.read_csv(filename2)
mu1 = np.array([1,1])
mu2 = np.array([3,2])
cov1 = np.array([[1, 0.3],
                [0.3, 1]])
cov2 = np.array([[1, 0.15],
                [0.15, 1]])
def two_dimensional_normal(x, mu, cov):
    diff = (x - mu)[:,np.newaxis]
    exponent = exp(-0.5 * diff.T @ inv(cov) @ diff)
    p = (1 / (2*pi * sqrt(det(cov)))) * exponent
    return p
def classify(data2,r):
    x = [0,0]
    c = [0,0]
    p = [0,0]
    c1_1 = []
    c1_2 = []
    c2_1 = []
    c2_2 = []
    for i in range(len(data2)):
        x[0] = data2.at[i,"x1"]
        x[1] = data2.at[i,"x2"]
        p[0] = two_dimensional_normal(x, mu1, cov1)
        p[1] = two_dimensional_normal(x, mu2, cov2)
        c[0] = r[0,1]*p[1]
        c[1] = r[1,0]*p[0]
        if c[0] < c[1]:
            c1_1.append(x[0])
            c1_2.append(x[1])
        else:
            c2_1.append(x[0])
            c2_2.append(x[1])
    plt.figure()
    plt.scatter(c1_1, c1_2, color = 'r')
    plt.scatter(c2_1, c2_2)
    plt.legend(('class1','class2'))
    
r = np.array([[0, 1],
                [1, 0]])
classify(data2,r)
r = np.array([[0, 0.1],
                [0.9, 0]])
classify(data2,r)
r = np.array([[0, 0.02],
                [0.98, 0]])
classify(data2,r)
r = np.array([[0, 0.9],
                [0.1, 0]])
classify(data2,r)
r = np.array([[0, 0.98],
                [0.02, 0]])
classify(data2,r)
