"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from sklearn.decomposition import SparsePCA, PCA
from utils.visualize import Visualizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import IPython
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, DBSCAN

x = np.load(f"../vectorize/path/~tienda1~publico~entrar.jsp_x.npy")
x = StandardScaler().fit_transform(x)
y = np.load(f"../vectorize/path/~tienda1~publico~entrar.jsp_y.npy")
index = np.load(f"../vectorze/path/~tienda1~publico~entrar.jsp_index.npy")
print(x.shape)
print(np.sum(y))
#for visualize
x_d = TSNE(n_components=2).fit_transform(x)
#x_d = PCA(n_components=2).fit_transform(x)

vis = Visualizer()
#vis.scatter(X=x_d, Y=y+1, opts={'markersize': 5})

def evaluate(clf, x_d, y, name):
    y_pred = clf.fit_predict(x_d)
    matrix = confusion_matrix(y, y_pred)
    TP, FP = matrix[0]
    FN, TN = matrix[1]
    PPV = (TP * 1.0) / (TP + FP)
    TPR = (TP * 1.0) / (TP + FN)
    TNR = (FP * 1.0) / (TN + FP)
    ACC = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    F1 = 2.0 * PPV * TPR / (PPV + TPR)
    print("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (name, PPV, TPR, TNR, ACC, F1))

'''
print ("Machine Learning Algorithms   \tPPV\t\tFPR\t\tTPR\t\tACC\t\tF1")
lof = LocalOutlierFactor(n_neighbors=30, n_jobs=-1)
isof = IsolationForest(n_jobs=-1)
ocsvm = OneClassSVM()
for clf in [lof, isof, ocsvm]:
    evaluate(clf, x_d, y, type(clf))
'''
for n_cluster in range(2,10):
    kmeans = KMeans(n_clusters=n_cluster)
    y_pred = kmeans.fit_predict(x)
    vis.scatter(X=x_d, Y=y_pred+1, opts={'title': f'KMeans-{n_cluster}', 'markersize': 5})

for eps in np.linspace(0.1, 3, 10):
    dbscan = DBSCAN(eps=eps)
    y_pred = dbscan.fit_predict(x)
    vis.scatter(X=x_d, Y=y_pred - np.min(y_pred) + 1, opts={'title': f'DBSCAN-{eps}', 'markersize': 5})
IPython.embed()
"""
xx, yy = np.meshgrid(np.linspace(-4, 4, 50), np.linspace(-4, 4, 50))
Z = lof._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

#a = plt.scatter(x_d[:-88, 0], x_d[:-88, 1], c='white', edgecolor='k', s=20)
#b = plt.scatter(x_d[-88:, 0], x_d[-88:, 1], c='red', edgecolor='k', s=20)
plt.scatter(x_d[:,0], x_d[:,1], c=y_pred, edgecolors='k', s=20)
plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
#plt.legend([a, b],
#           ["normal observations",
#            "abnormal observations"],
#           loc="upper left")
plt.show()
"""