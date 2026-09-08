#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:59:45 2026

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


# with parameter tuning 
# from sklearn.svm import SVC
# classifier = SVC() 
# classifier.fit(X_train, y_train) 


#with hyperparameter tuning 
from sklearn.svm import SVC
# classifier = SVC()
# classifier = SVC(C=10.0,kernel='sigmoid',gamma='auto') 
# classifier = SVC(C=1.0,kernel='poly',degree=3,gamma='scale') 
classifier = SVC(C=1.0,kernel='rbf',degree=3,gamma='scale') 
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

test case 1 -:
  testing -- 20% , without StandardScaler model 
  ac = 82.25 , bias = 76 , variance = 82 
  we need to comment the scaling code for this 
  classifier = SVC() #we need to use this for parameter tuning 
  
  
  
test case 2 -: 
    testing -- 25% , without StandardScaler model 
    ac = 80 , bias = 76  , variance = 80
    we need to comment the scaling code for this 
    classifier = SVC() #we need to use this for parameter tuning 
    
test case 3 -: 

    testing -- 20% , with StandardScaler
    ac = 95 , bias = 90 , variance = 95
    classifier = SVC() #we need to use this for parameter tuning 
    
test case 4 -:  with hyperparameter tuning we change system parameter 

   C REGULIRAZATION FOR OVERFIT 
   
   C= 10.0 
   kernal = 'sigmoid'
   gamma = 'auto'
   
   classifier = SVC(C=10.0,kernel='sigmoid',gamma='auto') 
    
   ac = 76 , bias = 65 , variance = 76 
   
   
test case 5 -: with hyperparameter tuning we change system parameter 
    C = 1.0
    kernal = 'poly'
    gamma = 'scale'
    degree = 3
    
    classifier = SVC(C=1.0,kernel='poly',degree=3,gamma='scale') 
    
    ac = 91 ,bais = 82 ,variance = 91
    
    
test case 6 -: 
    by default in svc algo 
    C = 1.0
    kernal = 'rbf'
    gamma ='scale'
    degree = 3
    
    so this is comes under the parameter tuning 
    when we chnage this like done in test case 5 test case 4 we change that is then 
    hyperparameter tuning
    
    ac = 95 , bias = 90 , variance = 95
    
    
"""
