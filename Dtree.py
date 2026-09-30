#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:34:51 2026

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
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier() 
classifier = RandomForestClassifier(n_estimators=30,random_state=0,criterion='gini'
                                    ,max_depth=10
                                    ) 

# classifier = DecisionTreeClassifier() 
# classifier = DecisionTreeClassifier(criterion='entropy',splitter='random',
#                                  max_depth=None,random_state=0,
#                                   ) 

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
    
"""







"""

Ranking table till 
Rank =      3   |  logistic regression -- AC = 91 
          2     |  SVM ---   ac = 95 , bias = 90 , variance = 95
         1      |  knn    ac = 95 , bias = 91 , variance = 95
                |  SSNBYS ac= 92 , bias = 87 , var = 92

"""