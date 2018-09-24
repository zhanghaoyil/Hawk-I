"""
Copyright (c) Zacker
Hawk I
"""

import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from utils.visualize import Visualizer

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

if __name__ == '__main__':
    vectors = np.load(f"../vectorize/paths/~tienda1~miembros~editar.jsp_x.npy")
    x = StandardScaler().fit_transform(vectors)
    y = np.load(f"../vectorize/paths/~tienda1~miembros~editar.jsp_y.npy")

    vis = Visualizer()
    for eps in np.linspace(0.4, 3.2, 8):
        clf = DBSCAN(eps=eps, min_samples=3)
        y_pred = clf.fit_predict(x)
        y_pred += (1 if -1 in y_pred else 1)
        counts = np.bincount(y_pred)
        n_clusters_ = len(set(y_pred))
        sorted_count_index = np.argsort(counts)
        all = len(y)
        axis_x = []
        axis_y = []
        for c in sorted_count_index:
            picked = counts[c]
            right = np.sum(y[y_pred == c] == 1)
            acc = right / picked
            ratio = picked / all
            axis_x.append(ratio)
            axis_y.append(acc)
        vis.line(X=axis_x, Y=axis_y, win=eps, opts={'title': f'EPS {round(eps, 2)} {n_clusters_} clusters', 'xlabel': 'cluster percentile', 'ylabel': 'anonymous sample percentile', 'markers': True, 'markersize': 5})