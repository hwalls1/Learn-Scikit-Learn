#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:21:07 2019

@author: harrisonwalls
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import LeaveOneOut


iris = load_iris()
predc = []
loo = LeaveOneOut()
loo.get_n_splits(iris.data)
clf = tree.DecisionTreeClassifier(criterion='gini')
for train_index, test_index in loo
    