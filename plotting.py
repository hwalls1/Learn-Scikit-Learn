import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


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
plt.savefig('fig1.pdf')

# change the color of the curve to black
fig = plt.figure()
plt.plot(y, 'k')
plt.savefig('fig2.png')

# plot markers of data points
fig = plt.figure()
plt.plot(y, '-ko')
plt.savefig('fig3.png')

# remove the curve
fig = plt.figure()
plt.plot(y, 'ko')
plt.savefig('fig4.png')


# remove the curve and change color to red
fig = plt.figure()
plt.plot(y, 'ro')
plt.savefig('fig5.png')

# change marker shape to triangle
fig = plt.figure()
plt.plot(y, 'g^')
plt.savefig('fig6.png')

# change marker shape to square
fig = plt.figure()
plt.plot(y, 'bs')
plt.savefig('fig7.png')

# Generating a 1D data and draw three different curves in the same figure
fig = plt.figure()
t = np.arange(0., 5.,0.2) # evenly sampled time at 200ms intervals
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3,'g^')
plt.savefig('fig8.png')


# Generating a 2D blob data and plot using plot function
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

fig = plt.figure()
plt.scatter(X[:,0],X[:,1],c = labels_true, marker='o')
plt.title('Blob data')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.savefig('fig9.png')


centers = [[1, 1], [-1, -1], [1, -1]]
labels_true = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
            memory=None, connectivity=None, compute_full_tree='auto',
            linkage='ward', pooling_func='deprecated').fit(X)

fig = plt.figure()
#lt.scatter(X[:,0],X[:,1],c = labels_true, marker='o')
plt.title('Blob data')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.savefig('fig10.png')
