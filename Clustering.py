#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:53:07 2020

@author: marcos
"""

print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from pandas import read_csv
from mpl_toolkits.mplot3d import Axes3D
# #############################################################################
# Generate sample data


# Load dataset
url = "data3.csv"

names = ['P.Altitude','IAS','TAT','TQ','Np','Nl','Nh','ITT','Wf']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(1000))
# descriptions
#print(dataset.describe())


array = dataset.values
X = array[:,0:8]
y = array[:,5:8]


# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=300)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()
plt.rcParams["figure.dpi"] = 600
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
i=1
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k

    cluster_center = cluster_centers[k]

    plt.plot(X[my_members, 0], X[my_members, 7], col + '.')
    plt.plot(cluster_center[0], cluster_center[7], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
    i=i+1
    print(X[my_members, 7])
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.grid(True)

plt.show()
