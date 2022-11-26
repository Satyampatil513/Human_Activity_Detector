# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 03:16:10 2022

@author: sattu
"""

# importing necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
#useless dropping
train.drop(['Unnamed: 0'], axis = 1, inplace = True)
test.drop(['Unnamed: 0'], axis = 1, inplace = True)
X_train = train.drop(['fall','label'],axis=1)
y_train = train['fall']
X_test =  test.drop(['fall','label'],axis=1)
y_test =  test['fall']

#after knowing this one is not useful
X_train.drop(['gyro_max'], axis=1, inplace=True)
X_test.drop(['gyro_max'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#rabdom forest
# Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [None,10,30,50,70]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 9, 12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100, cv = 5,
                               verbose=2, 
                               random_state=42, 
                               n_jobs = -1
                              )
rf_random.fit(X_train, y_train)
x=rf_random.predict(X_test)

Pkl_Filename = "random_forest.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random, file)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = (((predictions==test_labels).sum())/test_labels.shape[0])*100
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

best_estimator = rf_random.best_estimator_
optimal_accuracy = evaluate(best_estimator, X_test, y_test)