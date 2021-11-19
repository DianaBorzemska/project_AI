# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:38:25 2021

@author: diana
"""
from sklearn.model_selection import train_test_split
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import numpy as np

url = "Datasets/diabetesv2.csv"
names = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age', 'outcome']
dataset = read_csv(url, names=names)

array = dataset.values
X = array[:,0:8]
y = array[:,8]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model.fit(X_train, Y_train, epochs=150, batch_size=10, callbacks=[callback])

_, accuracy = model.evaluate(X_validation, Y_validation)
print('Accuracy: %.2f' % (accuracy*100))
"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)# path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model_diabetes.tflite', 'wb') as f:
  f.write(tflite_model)
  """