"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import time

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
    start = time.time()
    vectors = np.load(f"../vectorize/path-{time.strftime('%Y-%m-%d')}/~tienda1~miembros~editar.jsp_x.npy")
    targets = np.load(f"../vectorize/path-{time.strftime('%Y-%m-%d')}/~tienda1~miembros~editar.jsp_y.npy")
    print(vectors)
    print(vectors.shape, targets.shape)

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
    lr = LogisticRegression(penalty='l2')
    plot_learning_curve(lr, 'Hawk-I LR', MinMaxScaler().fit_transform(vectors), targets, cv=cv, n_jobs=4, train_sizes=np.linspace(0.1, 1, 10))
    rf = RandomForestClassifier(n_estimators=20, max_features=None)
    plot_learning_curve(rf, f'Hawk-I RF with 20 trees', MinMaxScaler().fit_transform(vectors), targets, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1, 5))

    svc = SVC()
    plot_learning_curve(svc, 'Hawk-I SVM', MinMaxScaler().fit_transform(vectors), targets, cv=cv, n_jobs=4, train_sizes=np.linspace(0.1, 1, 10))

    knn = KNeighborsClassifier(n_neighbors=10)
    plot_learning_curve(knn, 'Hawk-I KNN', MinMaxScaler().fit_transform(vectors), targets, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1, 5))


    end = time.time()
    print(f'Elapsed: {end-start}')
    plt.show()
