"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
from sklearn.externals import joblib
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

tfidf = joblib.load(f"../vectorize/path-2018-09-14/~tienda1~publico~pagar.jsp_tfidf.m")
print(tfidf)
IPython.embed()