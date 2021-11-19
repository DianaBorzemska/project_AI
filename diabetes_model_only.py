# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:23:02 2021

@author: diana
"""

#Diana Borzemska R00192880

from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn2pmml.decoration import ContinuousDomain

from sklearn.svm import SVC

from sklearn_pandas import DataFrameMapper



url = "Datasets/diabetesv2.csv"
names = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age', 'outcome']
dataset = read_csv(url, names=names)

default_mapper = DataFrameMapper([(names, [ContinuousDomain(), SimpleImputer()])])

array = dataset.values
X = array[:,0:8]
y = array[:,8]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
model = PMMLPipeline([ 
    ('mapper', default_mapper), 
    ('classifier', SVC(gamma='auto')) 
])


model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(f'The prediction accuracy is {accuracy_score(Y_validation, predictions)*100:.2f}')
sklearn2pmml(model, 'model_diabetes.pmml', with_repr = True)