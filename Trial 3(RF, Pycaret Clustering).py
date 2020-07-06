# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:35:24 2020

@author: niles
"""
import pandas as pd
import numpy as np
import matplotlib as plt
"""Setting up Environment in PyCaret"""
from pycaret.utils import version
version()

import pycaret
from pycaret.clustering import *

"""Reading the whole dataset CAL, CNC, GR HRD, HRM, PE, ZDEN, DTC, DTS"""
df = pd.read_csv('SWPLA_Dataset/train.csv')
# remove all rows that contains missing value
df.replace(['-999', -999], np.nan, inplace=True)
df.dropna(axis=0, inplace=True)

"""Outlier Removal"""
"""**Outliers removal based on thresholding**"""
df = df[(df.GR > 0) & (df.GR  <= 250)]
df = df[(df.PE > 0) & (df.PE  <= 8)]
df = df[(df.ZDEN > 1.95) & (df.ZDEN  <= 2.95)]
df = df[(df.CNC > 0) & (df.CNC  <= 1)]
df.shape

"""**Z-Score**"""
# Z-Score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df))
print(z)
threshold = 3
print(np.where(z > 3))
df = df[(z < 3).all(axis=1)]
df.shape

"""Reading the whole dataset CAL, CNC, GR HRD, HRM, PE, ZDEN, DTC, DTS"""
df1 = pd.read_csv('SWPLA_Dataset/train.csv')
# remove all rows that contains missing value
df.replace(['-999', -999], np.nan, inplace=True)
df.dropna(axis=0, inplace=True)

"""Outlier Removal"""
"""**Outliers removal based on thresholding**"""
df1 = df1[(df1.GR > 0) & (df1.GR  <= 250)]
df1 = df[(df1.PE > 0) & (df1.PE  <= 8)]
df1 = df[(df1.ZDEN > 1.95) & (df1.ZDEN  <= 2.95)]
df1 = df[(df1.CNC > 0) & (df1.CNC  <= 1)]
df1.shape

"""**Z-Score**"""
# Z-Score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df1))
print(z)
threshold = 3
print(np.where(z > 3))
df1 = df1[(z < 3).all(axis=1)]
df1.shape

"""Setting up model environment"""
cluster1 = setup(df.iloc[:,0:7], normalize = True, 
                   ignore_features = [],
                   session_id = 123)

# 7.0 Create a Model
kmeans = create_model('kmeans')
print(kmeans)
plot_model(kmeans)

"""We have created a kmeans model using `create_model()`. Notice the `n_clusters` parameter is set to `4` which is the default when you do not pass a value to the `num_clusters` parameter. In the below example we will create a `kmodes` model with 6 clusters."""

#kmodes = create_model('kmodes', num_clusters = 3)
#print(kmodes)

#dbscan = create_model('dbscan')
#dbscan.eps = 0.01
#dbscan.min_samples = 500
#print(dbscan)

hclust = create_model('hclust')
hclust.n_clusters =3
print(hclust)

birch = create_model('birch')
birch.n_clusters =3
birch

#optics = create_model('optics')
#optics.min_samples = 500
#optics.eps = 0.1

"""Simply replacing `kmeans` with `kmodes` inside `create_model()` has created a`kmodes` clustering model. There are 9 models available in the `pycaret.clustering` module. To see the complete list, please see the docstring. If you would like to read more about the use cases and limitations of different models, you may __[click here](https://scikit-learn.org/stable/modules/clustering.html)__ to read more.

# 8.0 Assign a Model

Now that we have created a model, we would like to assign the cluster labels to our dataset (1080 samples) to analyze the results. We will achieve this by using the `assign_model()` function. See an example below:
"""
kmean_results = assign_model(kmeans)
kmean_results.head()

# Encoding label data
kmean_results['Cluster'] = pd.Categorical(kmean_results['Cluster'])
kmean_resultsDummies = pd.get_dummies(kmean_results['Cluster'], prefix = '')
kmean_results = pd.concat([kmean_results, kmean_resultsDummies], axis=1)
kmean_results1 = kmean_results
#############################################################################
#############################################################################
# DTC
DTC_train = np.concatenate((kmean_results1.iloc[:,np.r_[0:7, 8:11]], df.iloc[:,7:9]),1)
DTC_train = pd.DataFrame(DTC_train, columns = ['CAL','CNC','GR','HRD','HRM','PE','ZDEN','Cluster 1','Cluster 2','Cluster 3','Cluster 4','DTC'])

from pycaret.regression import *
reg1 = setup(DTC_train, target = 'DTC', session_id = 123, silent = True) 
compare_models(blacklist = ['tr']) 

##############################################################################
##############################################################################
df1_train = df1.iloc[:, 0:7]

Cluster2 = setup(df1_train, normalize = True, 
                   ignore_features = [],
                   session_id = 123)
kmeans1 = create_model('kmeans')
print(kmeans1)

kmean_results1 = assign_model(kmeans1)
kmean_results1.head()

kmean_results1['Cluster'] = pd.Categorical(kmean_results1['Cluster'])
kmean_results1Dummies = pd.get_dummies(kmean_results1['Cluster'], prefix = '')
kmean_results1 = pd.concat([kmean_results1, kmean_results1Dummies], axis=1)
kmean_results1 = kmean_results1.iloc[:,np.r_[0:7, 8:12]]


