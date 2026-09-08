#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:08:53 2026

@author: sachinrajput
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/sachinrajput/Desktop/ml-learning/logit classification.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values 


# 20% dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20,
                                                   random_state=0)

# 25% dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                   test_size = 0.25,
#                                                  random_state=0)

# with scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier() 
classifier = KNeighborsClassifier(n_neighbors=8,p=1,algorithm='brute') 
classifier.fit(X_train, y_train) 

classifier.get_params()  

y_pred = classifier.predict(X_test)  


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac) 

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr) 

bias = classifier.score(X_train, y_train)
print(bias)

var = classifier.score(X_test, y_test)
print(var) 


"""
test case 1 -: with StandardScaler
    bydefault system parameter in knn
    
    
    classifier = KNeighborsClassifier() 
    
    
    {'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'n_jobs': None,
     'n_neighbors': 5,
     'p': 2,
     'weights': 'uniform'}
    ---know as parameter tuning if we chnages nothing-----
    
    where p =1 mean manhattan_distance use for farmost distance
    if p = 2 mean euclidean_distance
    
    result ----------------------------
    ac = 95 , bias = 91 , variance = 95
    -----------------------------------
    
test case 2 -: without StandardScaler
    when we do not apply without scalling 
    
    ac = 83 , bias = 87 , variance = 83
    
test case 3 -: with hyperparameter tuning and with StandardScaler
        
    hyperparameter 
    n_neighbors=8,
    p=1,
    algorithm='brute'
    
    
    classifier = KNeighborsClassifier(n_neighbors=8,p=1,algorithm='brute') 
    
    ac = 95 , bias = 90 , var = 95
    
"""


"""
Ranking table till 
Rank =      3   |  logistic regression -- AC = 91 
          2     |  SVM ---   ac = 95 , bias = 90 , variance = 95
         1        |  knn    ac = 95 , bias = 91 , variance = 95

"""