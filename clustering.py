from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets

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
    data.append([float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4]), float(toks[5]), float(toks[6]), float(toks[7]), float(toks[8]), float(toks[9]), float(toks[10]), float(toks[11]), float(toks[12]), float(toks[13])])
fin.close()

X = np.array(data)
loo = LeaveOneOut()
loo.get_n_splits(X)
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
result = metrics.silhouette_score(X, labels, metric='euclidean')
print("Coeffecient for k-means with wine data with leave one out: ", result)

min_k = 2
max_k = 5
for k in range(min_k, max_k + 1):
    X = np.array(data)

    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    result = metrics.silhouette_score(X, labels, metric='euclidean')
    print("Coeffecient for k-means with wine data: ", result, "k is: ", k)

# for Iris data
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target
loo = LeaveOneOut()
loo.get_n_splits(X)
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
result = metrics.silhouette_score(X, labels, metric='euclidean')
print("Coeffecient for k-means with iris dats is with leave one out: ", result)

min_k = 2
max_k = 5
for k in range(min_k, max_k + 1):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    result = metrics.silhouette_score(X, labels, metric='euclidean')
    print("Coeffecient for k-means with iris dats is: ", result, "k is: ", k)
