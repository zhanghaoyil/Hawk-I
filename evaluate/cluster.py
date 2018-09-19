"""
Copyright (c) Zacker
Hawk I
"""

import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

def cluster_scoring(estimator, X, y):
    y_pred = estimator.fit_predict(X)
    counts = np.bincount(y_pred)
    n_clusters_ = len(set(y_pred))
    min_half = np.argsort(counts)[:int(n_clusters_/2)]
    picked = 0
    right = 0
    for c in min_half:
        picked += counts[c]
        right += np.sum(y[y_pred==c] == 1)
    score = float(right / picked)
    return score

def analyse_cluster(estimator, X, y):
    y_pred = estimator.fit_predict(X)
    counts = np.bincount(y_pred)
    n_clusters_ = len(set(y_pred))
    for i in range(1, n_clusters_):
        min_i = np.argsort(counts)[:i]
        picked = 0
        right = 0
        for c in min_i:
            picked += counts[c]
            right += np.sum(y[y_pred==c] == 1)
        print(f'{n_clusters_}-kmeans min {i}: {float(right / picked)}')

if __name__ == '__main__':
    vectors = np.load(f"../vectorize/path-{time.strftime('%Y-%m-%d')}/~tienda1~publico~registro.jsp_x.npy")
    x = StandardScaler().fit_transform(vectors)
    y = np.load(f"../vectorize/path-{time.strftime('%Y-%m-%d')}/~tienda1~publico~registro.jsp_y.npy")

    for eps in np.linspace(0.1, 2, 10):
        clf = DBSCAN(eps=eps, min_samples=1)
        y_pred = clf.fit_predict(x)
        y_pred += (1 if -1 in y_pred else 0)
        counts = np.bincount(y_pred)
        n_clusters_ = len(set(y_pred))
        min_half = np.argsort(counts)[:n_clusters_]
        all = len(y)
        picked = 0
        right = 0
        for c in min_half:
            picked += counts[c]
            right += np.sum(y[y_pred == c] == 1)
            acc = right / picked
            ratio = picked / all
            print(f'DBSCAN eps {eps}: {acc}, {ratio}')
