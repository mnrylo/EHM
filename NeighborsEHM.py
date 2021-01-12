# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:24:16 2020

@author: marco
"""

# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from pandas import read_csv
from sklearn.model_selection import train_test_split

url = "data3.csv"

names = ['P.Altitude','IAS','TAT','TQ','Np','Nl','Nh','ITT','Wf']
dataset = read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
y = array[:,5:9]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, random_state=1)
x_vector=[]
x_temp=0
for xi, X in enumerate(X_train):
    x_temp=x_temp+1
    x_vector.append(x_temp)
x_temp1=0  
x_vector1=[]
for xi, X in enumerate(X_validation):
    x_temp1=x_temp1+1
    x_vector1.append(x_temp1)


# #############################################################################
# Fit regression model
n_neighbors = 3

for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X_train, Y_train).predict(X_validation)
        
        plt.subplot(2, 1, i + 1)
        plt.scatter(x_vector1, Y_validation[:,0], color='darkorange', label='data')
        plt.plot(x_vector1, y_[:,0], color='black', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                    weights))

plt.tight_layout()
plt.savefig('Plot%f.png')
plt.show()

""" 
 plt.title("Parameter:%s" %(names[8]))
 plt.scatter(x_vector1, Y_validation[:,2],color='black', label='Validation')
 plt.plot(x_vector1, y_[:,2],color='blue', label='Prediction')
 plt.legend()
 plt.axis('tight')
 plt.show()
 
 plt.title("Parameter:%s" %(names[7]))
 plt.scatter(x_vector1, Y_validation[:,1],color='black', label='Validation')
 plt.plot(x_vector1, y_[:,1],color='blue', label='Prediction')
 plt.legend()
 plt.axis('tight')
 plt.show()

 plt.title("Parameter:%s" %(names[6]))
 plt.scatter(x_vector1, Y_validation[:,0],color='black', label='Validation')
 plt.plot(x_vector1, y_[:,0],color='blue', label='Prediction')
 plt.legend()
 plt.axis('tight')
 plt.show()
 """


