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
from sklearn.cluster import KMeans
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
total_data_scaled = sc.fit_transform(total_data)

kmeans_X = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X)
kmeans_X_scaled = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X_scaled)
kmeans_total_data = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(total_data)
kmeans_total_data_scaled = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(total_data_scaled)

'''
align_clusters_labels(kmeans_X.labels_)
align_clusters_labels(kmeans_X_scaled.labels_)
align_clusters_labels(kmeans_total_data.labels_)
align_clusters_labels(kmeans_total_data_scaled.labels_)
'''

metrics_report = {'X':{},
                  'X_scaled':{},
                  'total_data':{},
                  'total_data_scaled':{},
                  }

labels = {'X':kmeans_X.labels_,
          'X_scaled':kmeans_X_scaled.labels_,
          'total_data':kmeans_total_data.labels_,
          'total_data_scaled':kmeans_total_data_scaled.labels_,
         }

cluster_centers = {'X':kmeans_X.cluster_centers_,
                  'X_scaled':kmeans_X_scaled.cluster_centers_,
                  'total_data':kmeans_total_data.cluster_centers_,
                  'total_data_scaled':kmeans_total_data_scaled.cluster_centers_,
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

#visualizing - clustering of X_scaled dataset
plt.scatter(X_scaled[kmeans_X_scaled.labels_ ==1,4], X_scaled[kmeans_X_scaled.labels_ == 1,8], s=20, c='blue', label = 'Cluster 1')
plt.scatter(X_scaled[kmeans_X_scaled.labels_ ==0,4], X_scaled[kmeans_X_scaled.labels_ == 0,8], s=20, c='red', label = 'Cluster 2')
plt.scatter(kmeans_X_scaled.cluster_centers_[:,0], kmeans_X_scaled.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('Clustering of Travel Insurance Dataset with no classification labels')
plt.xlabel('Duration')
plt.ylabel('Age')
plt.legend()
plt.show()


#visualizing - clustering of total_dataset_scaled 
plt.scatter(total_data_scaled[kmeans_total_data_scaled.labels_ ==1,4], total_data_scaled[kmeans_total_data_scaled.labels_ == 1,8], s=20, c='blue', label = 'Cluster 1')
plt.scatter(total_data_scaled[kmeans_total_data_scaled.labels_ ==0,4], total_data_scaled[kmeans_total_data_scaled.labels_ == 0,8], s=20, c='red', label = 'Cluster 2')
plt.scatter(kmeans_total_data_scaled.cluster_centers_[:,0], kmeans_total_data_scaled.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('Clustering of Travel Insurance Dataset with classification labels')
plt.xlabel('Duration')
plt.ylabel('Age')
plt.legend()
plt.show()


#Visualizing given dataset
plt.scatter(X[y ==1,4], X[y == 1,8], s=40, c='blue', label = 'Claim: No')
plt.scatter(X[y ==0,4], X[y == 0,8], s=40, c='red', label = 'Caim: Yes')
plt.title('Classification of Travel Insurance dataset as it is')
plt.xlabel('Duration')
plt.ylabel('Range')
plt.legend()
plt.show()



