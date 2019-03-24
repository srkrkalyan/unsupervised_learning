#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 19:52:13 2019

@author: kalyantulabandu
"""

"""
Spyder Editor

This is a temporary script file.
"""
# Importing libraries
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def align_clusters_labels(labels):
    for index in range(0,len(labels)):
        labels[index] = 1-labels[index]

# Importing the dataset
dataset = pd.read_csv('travel_insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
total_data = dataset.iloc[:,:].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:,0])

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

labelencoder_X_3 = LabelEncoder()
X[:,3] = labelencoder_X_3.fit_transform(X[:,3])

labelencoder_X_5 = LabelEncoder()
X[:,5] = labelencoder_X_5.fit_transform(X[:,5])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

total_data = np.c_[X,y]

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
total_data_scaled = np.c_[X_scaled,y]

gmm_X = GaussianMixture(n_components=2).fit_predict(X)
gmm_X_scaled = GaussianMixture(n_components=2).fit_predict(X_scaled)
gmm_total_data = GaussianMixture(n_components=2).fit_predict(total_data)
gmm_total_data_scaled = GaussianMixture(n_components=2).fit_predict(total_data_scaled)


#align_clusters_labels(gmm_X)
align_clusters_labels(gmm_X_scaled)
align_clusters_labels(gmm_total_data)
#align_clusters_labels(gmm_total_data_scaled)


metrics_report = {'X':{},
                  'X_scaled':{},
                  'total_data':{},
                  'total_data_scaled':{},
                  }

labels = {'X':gmm_X,
          'X_scaled':gmm_X_scaled,
          'total_data':gmm_total_data,
          'total_data_scaled':gmm_total_data_scaled,
         }

for each in metrics_report.keys():
    metrics_report[each]['ARI'] = round(metrics.adjusted_rand_score(y,labels[each]),2)
    metrics_report[each]['AMI'] = round(metrics.adjusted_mutual_info_score(y,labels[each]),2)
    metrics_report[each]['homogeneity'] = round(metrics.homogeneity_score(y,labels[each]),2)
    metrics_report[each]['completeness'] = round(metrics.completeness_score(y,labels[each]),2)
    metrics_report[each]['v_measure'] = round(metrics.v_measure_score(y,labels[each]),2)
    metrics_report[each]['silhouette'] = round(metrics.silhouette_score(X,labels[each]),2)
    metrics_report[each]['accuracy'] = round(metrics.accuracy_score(y,labels[each])*100,2)

print(metrics_report)

print(gmm_total_data_scaled)


#visualizing EM clustering on travel insurance data set
plt.scatter(X[gmm_X_scaled ==0,4], X[gmm_X_scaled == 0,8], s=40, c='blue', label = 'Cluster 1')
plt.scatter(X[gmm_X_scaled ==1,4], X[gmm_X_scaled == 1,8], s=40, c='red', label = 'Cluster 2')
plt.title('EM Clustering of Travel Insurance Dataset with no classification labels')
plt.xlabel('Duration')
plt.ylabel('Age')
plt.legend()
plt.show()


