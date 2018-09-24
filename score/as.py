"""
Copyright (c) Zacker
Hawk I
"""

import numpy as np
import json
from evaluate.gaussian import anomaly_detect
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    x = np.load(f"../vectorize/paths/~tienda1~publico~registro.jsp_x.npy")
    params = np.load(f"../vectorize/paths/~tienda1~publico~registro.jsp_params.npy")
    with open(f"../vectorize/paths/~tienda1~publico~registro.jsp_samples.json", 'r') as sf:
        samples = json.loads(sf.readline())

    ases = StandardScaler().fit_transform(x[:, :len(params)])
    indices = ases > 6

    #extract anomalous payload
    for s in range(indices.shape[0]):
        for p in range(indices.shape[1]):
            if indices[s, p] and params[p] in samples[s]['OriParams'].keys() and samples[s]['OriParams'][params[p]].strip():
                print(f"##{params[p]}## ##{samples[s]['OriParams'][params[p]]}##")
