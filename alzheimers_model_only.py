# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:21:50 2021

@author: diana
"""

#Diana Borzemska R00192880

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn.naive_bayes import GaussianNB


url = "Datasets/alzheimers.csv"
names = ['mri_delay', 'sex',
          'age', 'education_years','social_status', 'mental_state_exam', 'clinical_dementia_rating',
         'estimated_tot_intracranial_vol', 'norm_brain_vol', 'atlas_scaling_factor', 'diagnosis']
dataset = read_csv(url, names=names)

array = dataset.values

# Split-out validation dataset

X = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=2)
model = PMMLPipeline([ ('classifier', GaussianNB()) ])

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(f'The prediction accuracy is {accuracy_score(Y_validation, predictions)*100:.2f}')
#sklearn2pmml(model, 'model_alzheimers.pmml', with_repr = True)