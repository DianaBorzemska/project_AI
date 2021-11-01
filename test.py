# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:57:28 2021

@author: diana
"""

import diabetes_model as dm
import alzheimers_model as am
import heart_disease_model as hdm
from sklearn.svm import SVC


def testDm(dm):
    news = [[0,135,68,42,250,42.3,0.365,24]]
    model = SVC(gamma='auto')
    newpredict = model.predict(news)
    print(newpredict)
    
testDm(dm)