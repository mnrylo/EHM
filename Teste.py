# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:53:22 2020

@author: marco
"""

import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: {}'.format(scipy.__version__))
import numpy
print('numpy: {}'.format(numpy.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('pandas: {}'.format(pandas.__version__))
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from time import time



# Load dataset
url = "data3.csv"
names = ['P.Altitude','IAS','TAT','TQ','Np','Nl','Nh','ITT','Wf']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(141))
# descriptions
#print(dataset.describe())


array = dataset.values
X = array[:,0:4]
y = array[:,5:8]
X_train, X_Test, Y_train, Y_Test = train_test_split(X, y, test_size=0.10, random_state=1)
x_vector=[]
x_temp=0
for xi, X in enumerate(X_train):
    x_temp=x_temp+1
    x_vector.append(x_temp)
    
x_temp1=0  
x_vector1=[]
for xi, X in enumerate(X_Test):
    x_temp1=x_temp1+1
    x_vector1.append(x_temp1)

print("Training MLPRegressor...")
tic = time()
est = MLPRegressor(hidden_layer_sizes=(80), 
                   activation='relu',
                   solver='lbfgs', 
                   alpha=0.0001, 
                   batch_size='auto', 
                   learning_rate='adaptive', 
                   learning_rate_init=0.01, 
                   power_t=0.5, max_iter=5000, 
                   shuffle=True, random_state=None, 
                   tol=0.0001, verbose=False, 
                   warm_start=False, momentum=0.9, 
                   nesterovs_momentum=True, 
                   early_stopping=True, 
                   validation_fraction=0.1, 
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                   n_iter_no_change=10, max_fun=15000).fit(X_train, Y_train)

y_=est.predict(X_Test);

print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(X_Test, Y_Test)))

img01=pyplot.scatter(x_vector1, Y_Test[:,2], color='darkorange', label='data')
img01=pyplot.plot(x_vector1, y_[:,2], color='black', label='prediction')
img01=pyplot.axis('tight')
pyplot.show()
