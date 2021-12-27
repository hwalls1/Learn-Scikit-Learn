import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

"""
noisy cirlces 
"""
n_samples = 1000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=.05)

X = noisy_circles[0]
Y = noisy_circles[1]
X_norm = StandardScaler().fit_transform(X)
clustering = DBSCAN(eps=0.2, min_samples=2).fit(X_norm)

plt.figure()
plt.scatter(X_norm[:,0], X_norm[:,1], c=clustering.labels_, cmap=plt.cm.Set1, edgecolor='k')
plt.show()
plt.savefig('data noisy cirlces.png')



""""
Nosiy moons
"""
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.08)

X = noisy_moons[0]
Y = noisy_moons[1]

plt.figure()
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()
plt.savefig('data_noisy_moons.png')


"""
anisotropicly distributed data
"""
random_state = 170

X,Y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, Y)
X = aniso[0]
Y = aniso[1]
plt.figure()
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Set1, edgecolor='k')
plt.show()
plt.savefig('data_anisotropical_moons.png')


X_norm = StandardScaler().fit_transform(X)
clustering = DBSCAN(eps=0.2, min_samples=2).fit(X_norm)
