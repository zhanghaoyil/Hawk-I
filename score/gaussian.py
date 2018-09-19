"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
import scipy.stats as st
from utils.visualize import Visualizer

if __name__ == '__main__':
    x = np.load(f"../vectorize/path-2018-09-15/~tienda1~publico~registro.jsp_x.npy")
    y = np.load(f"../vectorize/path-2018-09-15/~tienda1~publico~registro.jsp_y.npy")
    y += 1
    gaussian_params = []

    #calculate gaussian distribution
    for v in range(x.shape[1]):
        mean = np.mean(x[:, v])
        std = np.std(x[:, v])
        gaussian_params.append((mean, std))

    scores = []
    for si in range(len(x)):
        sp = 1
        for v in range(len(gaussian_params)):
            p = st.norm.pdf(x[si][v], loc=gaussian_params[v][0], scale=gaussian_params[v][1])
            sp *= p
        scores.append([si, -np.log(sp)])

    vis = Visualizer()
    vis.scatter(X=scores, Y=y, opts={'markersize': 5})


