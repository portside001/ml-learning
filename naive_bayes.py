#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:46:33 2026

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

# with scaling with MultinomialNB apply this 
# from sklearn.preprocessing import Normalizer
# sc = Normalizer()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# with scaling 
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)




# from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
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
test case 1-:
    BernoulliNB with scalling 
    ac = 82 , bias = 70 , variance = 82


test case 2-: 
    BernoulliNB without scalling 
    ac = 72 , bias = 62 , var = 72 
    

test case 3 -: with out scalling 
    MultinomialNB
    ac = 56 , bias = 67 , var = 67 
    
test case 4 -: 
    to avoid negative sacalling we use normalize scalling  values will 
    be in between 0 and 1 
    
    
    ac = 72 , bais = 62 , var = 72
    
test case 5 -:
    GaussianNB  it does not required feture scalling 
    ac = 91 , bias = 88 , var = 91
    
    
test case 6 -:
    GaussianNB bell curve gaussion distribution
    ac= 92 , bias = 87 , var = 92
    
"""







"""

Ranking table till 
Rank =      3   |  logistic regression -- AC = 91 
          2     |  SVM ---   ac = 95 , bias = 90 , variance = 95
         1      |  knn    ac = 95 , bias = 91 , variance = 95
                |  SSNBYS ac= 92 , bias = 87 , var = 92

"""
