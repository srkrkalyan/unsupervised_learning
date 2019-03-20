#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing libraries
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np

def align_clusters_labels(labels):
    for index in range(0,len(labels)):
        labels[index] = 1-labels[index]

# Importing the dataset
dataset = pd.read_csv('breast_cancer_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
total_data = dataset.iloc[:,:].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
total_data_scaled = sc.fit_transform(total_data)

#total_data = np.c_[X,y]


'''

for components in range(1,X_scaled.shape[-1]+1):
    dataset= X_scaled
    pca = PCA(n_components = components)
    dataset = pca.fit_transform(dataset)
    #temp_y = y
    #align_clusters_labels(temp_y)
    kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(dataset)
    print(dataset.shape)
    print(1-accuracy_score(y,kmeans.labels_))
    dataset = np.c_[dataset,y]
    kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(dataset)
    #print(dataset.shape)
    print(1-accuracy_score(y,kmeans.labels_))
 '''   
    

X_transformed = GaussianRandomProjection(n_components=3).fit_transform(X_scaled)
X_transformed_y = np.c_[X_transformed,y]

'''
kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X_transformed)
align_clusters_labels(kmeans.labels_)
#print(dataset.shape)
print(accuracy_score(y,kmeans.labels_))
'''

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.3,random_state=1)

#Importing keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Add input layer and first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim =  3))

# Adding the second hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to training set
history = classifier.fit(X_transformed_train, y_train, validation_split = 0.33, batch_size = 10, nb_epoch = 120)

# Predicting the Test set results
y_pred = classifier.predict(X_transformed_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# comparing predictions and targets to compute accuracy
print("Artificial Neural Network - accuracy - Breast Cancer Prediction: ", round(metrics.accuracy_score(y_test,y_pred),6))


'''
# Clustering of X vector 
kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X)

#calculate accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,kmeans.labels_)
print(accuracy)
print(kmeans.get_params)
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)

#visualizing given data set
plt.scatter(X[kmeans.labels_ ==0,0], X[kmeans.labels_ == 0,1], s=20, c='red', label = 'Cluster 1')
plt.scatter(X[kmeans.labels_ ==1,0], X[kmeans.labels_ == 1,1], s=20, c='blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('Clustering of Travel Insurance Dataset with no classification labels')
plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.legend()
plt.show()


# Clustering of total dataset including classification label
kmeans_total = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(total_data)

# Calculate accuracy score
accuracy_totaldata = accuracy_score(y,kmeans_total.labels_)
print(1-accuracy_totaldata)

#visualizing clustering of given dataset
plt.scatter(total_data[kmeans_total.labels_ ==0,0], total_data[kmeans_total.labels_ == 0,1], s=40, c='red', label = 'Cluster 1')
plt.scatter(total_data[kmeans_total.labels_ ==1,0], total_data[kmeans_total.labels_ == 1,1], s=40, c='blue', label = 'Cluster 2')
plt.scatter(kmeans_total.cluster_centers_[:,0], kmeans_total.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('Clustering of Breast Cancer Dataset - With Classification Labels')
plt.xlabel('mean_radius')
plt.ylabel('mean_texture')
plt.legend()
plt.show()


#Visualizing given dataset
plt.scatter(X[y ==0,0], X[y == 0,1], s=20, c='red', label = 'Diagnosis: 0')
plt.scatter(X[y ==1,0], X[y == 1,1], s=20, c='blue', label = 'Diagnosis: 1')
plt.title('Classification of Travel Insurance dataset as it is')
plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.legend()
plt.show()
'''


