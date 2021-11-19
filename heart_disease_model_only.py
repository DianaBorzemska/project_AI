# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:23:38 2021

@author: diana
"""

#Diana Borzemska R00192880

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as ss



url = "Datasets/heart_diseasev2.csv"
names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestrol', 'fasting_blood_sugar',
         'resting_ecg', 'max_heart_rate', 'excercise_induced_angina','st_depression_exercise_induced', 'peak_ex_st', 'major_vessels_num',
         'thalassemia', 'diagnosis']
dataset = read_csv(url, names=names)

dataset['diagnosis'] = dataset.diagnosis.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})


array = dataset.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=3)

sc = ss()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)
model = PMMLPipeline([ ('classifier', SVC(gamma='auto')) ])


model.fit(X_train, Y_train)



predictions = model.predict(X_validation)

print(f'The prediction accuracy is {accuracy_score(Y_validation, predictions)*100:.2f}')
#sklearn2pmml(model, 'model_heart_disease.pmml', with_repr = True)
