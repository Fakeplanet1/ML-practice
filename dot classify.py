# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:23:02 2022

@author: Yen-Chen Lai
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pc

### read the file ###
filename = './dot_data.csv'
data = pd.read_csv(filename)

### pre-processing ###
class1 = data[data["class"] == 1]
class0 = data[data["class"] == 0]
class1.reset_index(inplace=True,drop=True)
class0.reset_index(inplace=True,drop=True)
avr1 = data['feature 1'].mean()
avr2 = data['feature2'].mean()
a = class1['feature 1'].min()
b = class1['feature 1'].max()
c = class1['feature2'].min()
d = class1['feature2'].max()
r_big = 100
r_small = 0

for i in range(len(class1)):
    x2 = class1.at[i,"feature 1"]
    y2 = class1.at[i,"feature2"]
    dis2 = (x2-avr1)**2 + (y2-avr2)**2
    if dis2 > r_small**2:
        r_small = dis2**0.5 
    x = class0.at[i,"feature 1"]
    y = class0.at[i,"feature2"]
    dis = (x-avr1)**2 + (y-avr2)**2    
    if dis < r_big**2:
        r_big = dis**0.5
        
up = []
down = []
right = []
left = []

for i in range(len(class1)):
    X = class0.at[i,"feature 1"]
    Y = class0.at[i,"feature2"]
    L1 = (X-a)*(d-c)/(b-a)+c
    L2 = (X-b)*(c-d)/(b-a)+c
    if Y < L1 and Y < L2:
        down.append(Y)
    elif Y >= L1 and Y < L2:
        right.append(X)
    elif Y >= L1 and Y >= L2:
        up.append(Y)
    else:
        left.append(X)
        
up.sort()
left.sort()
down.sort(reverse = True)
right.sort(reverse = True)

### plot small and big rectangles ###
plt.rcParams["figure.figsize"] = (6, 6)
ax = class1.plot(x = 'feature 1',y = 'feature2',kind = 'scatter',color = 'r',marker = '^', zorder = 2)
class0.plot(x = 'feature 1',y = 'feature2',kind = 'scatter', ax = ax)
rec_small = pc.Rectangle((a,c), b-a, d-c, color = 'lightgreen', zorder=1)   
ax.add_patch(rec_small)
rec_big = pc.Rectangle((left[0],down[0]), right[0]-left[0], up[0]-down[0], color = 'lightblue', zorder=0)   
ax.add_patch(rec_big)
plt.legend(('small','big','class1','class0'))

### plot small and big circles ###
ax2 = class1.plot(x = 'feature 1',y = 'feature2',kind = 'scatter',color = 'r',marker = '^')
class0.plot(x = 'feature 1',y = 'feature2',kind = 'scatter',ax = ax2)
cir_big = pc.Circle((avr1,avr2),r_big, color = 'lightblue', zorder = 0) 
ax2.add_artist(cir_big)
cir_small = pc.Circle((avr1,avr2),r_small, color = 'lightgreen', zorder = 0) 
ax2.add_artist(cir_small)
plt.legend(('class1','class0'))

### plot ellipses ###
ax3 = class1.plot(x = 'feature 1',y = 'feature2',kind = 'scatter',color = 'r',marker = '^')
class0.plot(x = 'feature 1',y = 'feature2',kind = 'scatter',ax = ax3)
e = pc.Ellipse(xy = (avr1,avr2), width = (r_big + r_small)/2*2, height = (2*r_small + r_big)/3*2, angle=0, color = 'lightgreen', zorder = 0)
ax3.add_artist(e)
plt.legend(('class1','class0'))

plt.show()