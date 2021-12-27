#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:53:57 2019

@author: harrisonwalls
"""

from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing

# step1. read the input file
data = []
clabels = []
fin = open("wine.data","r")
for str in fin:
    # skip empty lines
    if str == "\n":
        continue
    
    # retrieves features
    toks = str.split(",")
    clabels.append(toks[0])
    f4toks = toks[13].split("\n")
    data.append(int([toks[1]), float(toks[2]), float(toks[3]), float(toks[4]), float(toks[5]), float(toks[6]), float(toks[7]), float(toks[8]), float(toks[9]), float(toks[10]), float(toks[11]), float(toks[12]), float(toks[13])])
fin.close()

X = np.array(data)
le = preprocessing.LabelEncoder()
le.fit(clabels)
X_labels = le.transform(clabels)
        
# step2. train the GaussianNB and predict the test data.
# calculate the prediction accuracy using the leave-one-out cross-validation.
predc = []
loo = LeaveOneOut()
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    gnb = MultinomialNB()
    pc = gnb.fit(X_train, X_labels[train_index]).predict(X_test)
    predc.append(pc[0])

hits = 0
for i in range(len(X)):
    if X_labels[i] == predc[i]:
        hits = hits + 1
print("The prediction accuracy is ",hits ,"/",len(X),"=",hits/len(X))