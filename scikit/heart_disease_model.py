# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:08:10 2021

@author: diana
"""

#Diana Borzemska R00192880

from pandas import read_csv
#from pandas.plotting import scatter_matrix
#from matplotlib import pyplot
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as ss



url = "Datasets/heart_diseasev2.csv"
names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholestrol', 'fasting_blood_sugar',
         'resting_ecg', 'max_heart_rate', 'excercise_induced_angina','st_depression_exercise_induced', 'peak_ex_st', 'major_vessels_num',
         'thalassemia', 'diagnosis']
dataset = read_csv(url, names=names)

dataset['diagnosis'] = dataset.diagnosis.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

#sex=> 1 male 0 female
array = dataset.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=3)

sc = ss()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)


model = SVC(gamma='auto')
model.fit(X_train, Y_train)



predictions = model.predict(X_validation)

print(f'The prediction accuracy is {accuracy_score(Y_validation, predictions)*100:.2f}')




"""
#Input data prediction example
news = [[63,1,1,145,233,1,2,150,0,2.3,3,0,6]]
newpredict = model.predict(news)
print(newpredict)

"""
"""

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
"""


"""
Statistics
#shape
print(dataset.shape)

#head
print(dataset.head(20))

#dataset descriptions
print(dataset.describe())

#num of rows by class variable
print(dataset.groupby('diagnosis').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()"""