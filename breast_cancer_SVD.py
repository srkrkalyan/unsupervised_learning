#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing libraries
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
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


svd = TruncatedSVD(n_components=3)
X_scaled_transformed = svd.fit_transform(X_scaled)
X_scaled_transformed_y = np.c_[X_scaled_transformed,y]

print(svd.explained_variance_)
print(svd.explained_variance_ratio_)

kmeans = KMeans(n_clusters=2, random_state = 0, n_init=20).fit(X_scaled_transformed)
align_clusters_labels(kmeans.labels_)
gmm = GaussianMixture(n_components=2).fit_predict(X_scaled_transformed)
align_clusters_labels(gmm)


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

#visualizing - k-means clustering of SVD transformed dataset
plt.scatter(X_scaled_transformed[kmeans.labels_ ==1,0], X_scaled_transformed[kmeans.labels_ == 1,1], s=40, c='red', label = 'Cluster 1')
plt.scatter(X_scaled_transformed[kmeans.labels_ ==0,0], X_scaled_transformed[kmeans.labels_ == 0,1], s=40, c='blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title('k-means clustering on SVD transformed dataset - Breast Cancer Dataset')
plt.xlabel('SVD 0')
plt.ylabel('SVD 1')
plt.legend()
plt.show()

#visualizing - EM clustering of SVD transformed dataset
plt.scatter(X_scaled_transformed[gmm ==1,0], X_scaled_transformed[gmm == 1,1], s=40, c='red', label = 'Cluster 1')
plt.scatter(X_scaled_transformed[gmm ==0,0], X_scaled_transformed[gmm == 0,1], s=40, c='blue', label = 'Cluster 2')
plt.title('EM clustering on SVD transformed dataset - Breast Cancer Dataset')
plt.xlabel('SVD 0')
plt.ylabel('SVD 1')
plt.legend()
plt.show()

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_scaled_transformed,y,test_size=0.3,random_state=1)

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


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('ANN Accuracy - Breast Cancer SVD Data Set')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ANN Loss - Breast Cancer SVD Data Set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


