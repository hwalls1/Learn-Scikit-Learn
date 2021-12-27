from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

import random as random
from sklearn.datasets.samples_generator import make_blobs


centers = [[1,1],[-1,-1], [1,-1]]
X,labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

unique_labels = set(labels_true)
colors = [plt.cm.Spectral(each)
                for each in np.linspace(0,1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels_true == k)

    xy = X[class_member_mask]
    plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=tuple(col), markeredgecolor = 'k', markersize=14)

plt.title('blob data')
#plt.show()

"""
lab 2
"""
from mpl_toolkits.mplot3d import Axes3D

centers_3d = [[1,1,1], [-1,-1,-1], [1,-1,-1]]
X_3d, labels_true_3d = make_blobs(n_samples=750, centers=centers_3d, cluster_std = 0.4, random_state=0)

unique_labels_3d = set(labels_true_3d)
colors_3d = [plt.cm.Spectral(each)
            for each in np.linspace(0,1, len(unique_labels_3d))]
fig = plt.figure(1,figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_3d[:,0], X_3d[:, 1], X_3d[:,2], c=labels_true_3d, cmap=plt.cm.Set1, edgecolor='k', s=40)

ax2 = fig.gca(projection="3d")
ax2.view_init(azim=15)

ax.set_title("blob 3D data")
ax.set_xlabel("feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

#plt.show()


"""
lab 3
"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def myZscore(arr):
    norm_X = np.empty([len(arr), len(arr[0])])
    for i in range(len(arr[0])):
        mean_ = np.mean(arr[:,i], axis = 0)
        std_ = np.std(arr[:,i], axis = 0)
        norm_X[:,i] = (arr[:, i] - mean_)/ std_



A = normalize(X)
clustering = AgglomerativeClustering().fit(A)
clustering.labels_

pw = pairwise_distances(A)
Z = hierarchy.linkage(pw, 'single')

plt.figure()
dn = hierarchy.dendrogram(Z)

hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z, ax=axes[1],
                           above_threshold_color='#bcbddc',
                           orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
#plt.show()

"""
Lab 4
"""
"""
data = []
clabels = []
fdic = {"iris-setoa":0, "iris-versicolor": 1, "iris-virginica" :0 }
fin = open("/Users/harrisonwalls/Desktop/Machine_learning/iris.data.txt", "r")
for str in fin:
    if str == "\n":
        continue
    toks = str.split(",")
    clabels.append(fdic[cltoks[0]])
fin.close()

X = np.array(data)
X = myZscore(X)

xdist = mt.pairwise_distances(X, Y=None, metric = 'euclidean')
Z = hierachy.linkage(xdist, 'single')

plt.figure()
dn = hierachy.dendrogram(Z)

clustering = AgglomerativeClustering(affinitiy= 'euclidean', compute_full_tree='auto', connectivity=None,
                            linkage= 'complete', memory=None, n_clustering=3).fit(X)
print(clustering.labels_)

"""



"""
Generates a 1D sample data and plot curves
"""


numdp = 50
maxval = 50

y = []

for i in range(numdp):
    y.append(random.uniform(1, maxval))

plt.plot(y)
plt.xlabel("X label")
plt.ylabel("Y label")
plt.title("random title")
plt.savefig('fig1.png')
plt.savegfig('fig1.pdf')
