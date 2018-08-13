# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:00:37 2018

@author: Aafreen Dabhoiwala
"""


## Artificial Neural Network

# Installing Theano- Its an open source numerical computational library. In Theano, computation
#--- are expressed using a NumPy-esque syntax.
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing TensorFlow - TensorFlow is an open-source software library for dataflow programming 
#--across a range of tasks. It is a symbolic math library, and is also used for machine learning 
#---applications such as neural networks.

# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

#installing Keras-  Its an open source neural network library capable of running on top of
# --- TensorFlow , Theano

# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Aafreen Dabhoiwala/Documents/udemy/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical independent data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import tensorflow
import keras
# Sequential is used to initialize the neural network
from keras.models import Sequential

# Dense is used to create layers of the neural network
from keras.layers import Dense

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#----- 84.1% accuracy------------------------------

