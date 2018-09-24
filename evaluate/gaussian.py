"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
import scipy.stats as st
from utils.visualize import Visualizer

def anomaly_detect(x):
    gaussian_params = []

    #calculate gaussian distribution
    for v in range(x.shape[1]):
        mean = np.mean(x[:, v])
        std = np.std(x[:, v])
        gaussian_params.append((mean, std + 1e-3))  #with std smooth

    scores = []
    for si in range(len(x)):
        sp = 1
        for v in range(len(gaussian_params)):
            p = st.norm.pdf(x[si][v], loc=gaussian_params[v][0], scale=gaussian_params[v][1])
            sp *= p
        scores.append(-np.log(sp))

    return np.array(scores)

if __name__ == '__main__':
    x = np.load(f"../vectorize/paths/~tienda1~miembros~editar.jsp_x.npy")
    y = np.load(f"../vectorize/paths/~tienda1~miembros~editar.jsp_y.npy")
    abnormal_count = np.sum(y)
    y += 1

    scores = anomaly_detect(x)
    normal_max = -35
    ratio = float(np.sum(scores > normal_max) / abnormal_count)
    print(f'Distinguished ratio: {ratio}')

    scores = list(zip(range(len(y)), scores))
    vis = Visualizer()
    vis.scatter(X=scores, Y=y, opts={'markersize': 5, 'xlabel': 'samples', 'ylabel': 'Sample Anomaly Score'})


