#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:49:13 2019

@author: harrisonwalls
"""

from sklearn import tree
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
import graphviz

# step1. read the input file
data = []
clabels = []
fin = open("balance-scale.data","r")
for str in fin:
    # skip empty lines
    if str == "\n":
        continue

    toks = str.split(",")
    clabels.append(toks[0])
    f4toks = toks[4].split("\n")
    data.append([int(toks[1]),int(toks[2]),int(toks[3]),int(f4toks[0])])
fin.close()

X = np.array(data)
le = preprocessing.LabelEncoder()
le.fit(clabels)
X_labels = le.transform(clabels)
        
# step2. train the tree and predict the test data.
# calculate the prediction accuracy using the leave-one-out cross-validation.
predc = []
loo = LeaveOneOut()
loo.get_n_splits(X)
clf = tree.DecisionTreeClassifier(criterion="entropy")
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    pc = clf.fit(X_train, X_labels[train_index]).predict(X_test)
    predc.append(pc[0])

hits = 0
for i in range(len(X)):
    if X_labels[i] == predc[i]:
        hits = hits + 1
print("The prediction accuracy is ",hits ,"/",len(X),"=",hits/len(X))

feature_names = ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
target_names = ['B', 'R', 'L']
dot_data = tree.export_graphviz(clf, out_file='None', feature_names=feature_names,
                                class_names=target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("outpt")
