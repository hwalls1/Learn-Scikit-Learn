import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets


def digits():
    X, y = load_digits(return_X_y=True)
    predc = []
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    clf = Perceptron(max_iter = 100, random_state=0, tol=1e-3)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predc.append(clf.predict(X_test)[0])

    print(clf.score(X, y))

def myPerceptron(X,y,name):
    predc = []
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    clf = Perceptron(max_iter = 300, random_state=0, tol=1e-3)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predc.append(clf.predict(X_test)[0])

    print('accuracy', name, clf.score(X, y))

    fig = plt.figure()
    plt.scatter(X[:,0],X[:,1],c = predc, marker='o')
    plt.title(name)
    plt.savefig(name)

def main():
    X, labels_true = make_blobs(n_samples=500, centers=3, n_features = 2,
                                random_state=0)

    noisy_circles = datasets.make_circles(n_samples=100, factor=0.5, noise=.05)
    X_circles = noisy_circles[0]
    Y_circles = noisy_circles[1]

    noisy_moons = datasets.make_moons(n_samples=500, noise=.08)
    X_moons = noisy_moons[0]
    Y_moons = noisy_moons[1]

    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, labels_true)
    X_aniso = aniso[0]
    Y_aniso = aniso[1]
    myPerceptron(X, labels_true, 'blob.png')
    myPerceptron(X_circles, Y_circles, 'circles.png')
    myPerceptron(X_moons, Y_moons, 'moons.png')
    myPerceptron(X_aniso, Y_aniso, 'aniso.png')

    #digits()




if __name__ == '__main__':
    main()
