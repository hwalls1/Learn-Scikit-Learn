import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut

# step1. read the input file.
data = []
clabels = []
fdic = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
fin = open("iris.data","r")
for str in fin:
    # skip empty lines
    if str == "\n":
        continue

    # retrieves features
    toks = str.split(",")
    data.append([float(toks[0]),float(toks[1]),float(toks[2]),float(toks[3])])

    # retrieves class labels
    cltoks = toks[4].split("\n")
    clabels.append(fdic[cltoks[0]])
fin.close()

# step2. train the GaussianNB and predict the test data.
# calculate the prediction accuracy using the leave-one-out cross-validation.
predc = []
X = np.array(data)
X_labels = np.array(clabels)
loo = LeaveOneOut()
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    gnb = GaussianNB()
    pc = gnb.fit(X_train, X_labels[train_index]).predict(X_test)
    predc.append(pc[0])

hits = 0
for i in range(len(X)):
    if clabels[i] == predc[i]:
        hits = hits + 1
print("The prediction accuracy is ",hits ,"/",len(X),"=",hits/len(X))
