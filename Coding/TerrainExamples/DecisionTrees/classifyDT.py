# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 06:09:20 2015

@author: Patrick
"""
from sklearn import tree

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    
    return clf