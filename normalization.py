#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:18:13 2019

@author: harrisonwalls
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing

# convert a list of categorical values to a numerical representation
#def encodeList(mdarray):    
#    X = np.empty([len(mdarray), len(mdarray[0])])
#    le = preprocessing.LabelEncoder()
#    
#    if len(mdarray[0]) == 1:
#        le.fit(mdarray)
#        X = le.transform(mdarray)
#        return X
#    
#    for i in range(len(mdarray[0])):
#        le.fit(mdarray[:,i])
#        lst = le.transform(mdarray[:,i])
#        X[:,i] = lst
#        
#    return X

# step1. read the input file.
data = []
clabels = []
fin = open("/Users/harrisonwalls/Desktop/Machine_learning/wine.data","r")
for str in fin:
    # skip empty lines
    if str == "\n":
        continue
    
    # retrieves features
    toks = str.split(",")
    data.append([toks[0],toks[1],toks[2],toks[3], toks[4], toks[5], toks[6], toks[7], toks[8], toks[9], toks[10], toks[11], toks[12], toks[13], toks[14]])
    
    # retrieves class labels
    cltoks = toks[14].split("\n")
    
    clabels.append(cltoks[0])
fin.close()

# step 2. Encode categorical attributes and class labels to numbers. 
#X = encodeList(np.array(data))
#X_labels = encodeList(np.array(clabels))

# step3. train the GaussianNB and predict the test data.
# calculate the prediction accuracy using the leave-one-out cross-validation.
predc = []
loo = LeaveOneOut()
loo.get_n_splits(X)
clf = BernoulliNB()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    pc = clf.fit(X_train, X_labels[train_index])
    predc.append(clf.predict(X_test))
    
hits = 0
for i in range(len(X)):
    if X_labels[i] == predc[i]:
        hits = hits + 1
print("The prediction accuracy is ",hits ,"/",len(X),"=",hits/len(X))
    



