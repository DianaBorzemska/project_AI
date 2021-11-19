# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:38:53 2021

@author: diana
"""

from sklearn.model_selection import train_test_split
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

url = "Datasets/heart_diseasev2.csv"
names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestrol', 'fasting_blood_sugar',
         'resting_ecg', 'max_heart_rate', 'excercise_induced_angina','st_depression_exercise_induced', 'peak_ex_st', 'major_vessels_num',
         'thalassemia', 'diagnosis']
dataset = read_csv(url, names=names)

dataset['diagnosis'] = dataset.diagnosis.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})


array = dataset.values
X = array[:,0:-1]
y = array[:,-1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = Sequential()
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
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
with open('model_heart.tflite', 'wb') as f:
  f.write(tflite_model)
  """