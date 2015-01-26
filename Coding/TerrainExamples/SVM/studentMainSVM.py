# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:54:24 2015

@author: Patrick
"""

#!/usr/bin/python

""" Complete the code below with the sklearn Naive Bayes
    classifier to classify the terrain data
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



#clf = classify(features_train, labels_train)
#GaussianNB()
#predlabels = clf.predict(features_test)

clf = svm.SVC(gamma=1, C=1000, kernel='rbf');
clf.fit(features_train, labels_train);
predlabels = clf.predict(features_test)
"""
from sklearn.svm import SVC
SVC.__init__(C=1000, cache_size=200, class_weight=None, coef0=0.0, degree=3
    , gamma = 1000.0, kernel='rbf', max_iter = -1, probability = False
    , random_state = None, shrinking=True, tol=0.001, verbose=False);
"""

    ### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('test.png');
imgplot = plt.imshow(img);
plt.show();
#this should work, but it doesn't for some reason... 








