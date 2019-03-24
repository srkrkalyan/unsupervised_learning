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
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import time

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


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
total_data_scaled = np.c_[X_scaled,y]


X_scaled_transformed = GaussianRandomProjection(n_components=3, random_state=2).fit_transform(X_scaled)
X_scaled_transformed_y = np.c_[X_scaled_transformed,y]


kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X_scaled_transformed)
#align_clusters_labels(kmeans.labels_)
gmm = GaussianMixture(n_components=2).fit_predict(X_scaled_transformed)
#align_clusters_labels(gmm)

metrics_report = {'kmeans':{},
                  'gmm':{}
                  }

labels = {'kmeans':kmeans.labels_,
          'gmm':gmm
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

#visualizing - k-means clustering of PCA transformed dataset
plt.scatter(X_scaled_transformed[kmeans.labels_ ==1,0], X_scaled_transformed[kmeans.labels_ == 1,1], s=40, c='red', label = 'Cluster 1')
plt.scatter(X_scaled_transformed[kmeans.labels_ ==0,0], X_scaled_transformed[kmeans.labels_ == 0,1], s=40, c='blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('k-means clustering on ICA transformed dataset - Travel Insurance Dataset')
plt.xlabel('RP 0')
plt.ylabel('RP 1')
plt.legend()
plt.show()

#visualizing - EM clustering of PCA transformed dataset
plt.scatter(X_scaled_transformed[gmm ==1,0], X_scaled_transformed[gmm == 1,1], s=40, c='red', label = 'Cluster 1')
plt.scatter(X_scaled_transformed[gmm ==0,0], X_scaled_transformed[gmm == 0,1], s=40, c='blue', label = 'Cluster 2')
plt.title('EM clustering on ICA transformed dataset - Travel Insurance Dataset')
plt.xlabel('RP 0')
plt.ylabel('RP 1')
plt.legend()
plt.show()


#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense


# ANN run on feature transformed data set

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_scaled_transformed,y,test_size=0.3,random_state=1)

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

start_time = time.time()
# Fitting the ANN to training set
history = classifier.fit(X_transformed_train, y_train, batch_size = 1000, nb_epoch = 160, validation_split = 0.33)

print("--- %s seconds ---" % (time.time() - start_time))

# Predicting the Test set results
y_pred = classifier.predict(X_transformed_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("ANN model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),6))


#plotting learning curves for neural network
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ANN Accuracy - Travel Insurance Data Set')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss - Travel Insurance Data Set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ANN run on transformed data set + k-means clustering labels

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_scaled_transformed,kmeans.labels_,test_size=0.3,random_state=1)

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

start_time = time.time()
# Fitting the ANN to training set
history = classifier.fit(X_transformed_train, y_train, batch_size = 1000, nb_epoch = 160, validation_split = 0.33)

print("--- %s seconds ---" % (time.time() - start_time))

# Predicting the Test set results
y_pred = classifier.predict(X_transformed_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("ANN model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),6))


#plotting learning curves for neural network
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ANN Accuracy - Travel Insurance Data Set')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss - Travel Insurance Data Set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# ANN run on transformed data set + EM clustering labels

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_scaled_transformed,gmm,test_size=0.3,random_state=1)

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

start_time = time.time()
# Fitting the ANN to training set
history = classifier.fit(X_transformed_train, y_train, batch_size = 1000, nb_epoch = 160, validation_split = 0.33)

print("--- %s seconds ---" % (time.time() - start_time))

# Predicting the Test set results
y_pred = classifier.predict(X_transformed_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("ANN model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),6))


#plotting learning curves for neural network
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ANN Accuracy - Travel Insurance Data Set')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss - Travel Insurance Data Set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
