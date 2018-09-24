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
    sases = StandardScaler().fit_transform(X=anomaly_detect(x).reshape(-1, 1))

    params = np.load(f"../vectorize/paths/~tienda1~publico~registro.jsp_params.npy")

    with open(f"../vectorize/paths/~tienda1~publico~registro.jsp_samples.json", 'r') as sf:
        samples = json.loads(sf.readline())
    pases = StandardScaler().fit_transform(x[:, :len(params)])
    print(np.min(pases), np.max(pases), np.mean(pases))
    indices = pases > 6
    '''
    ases = []
    for i in range(len(x)):
        ases.append(pases[i] * sases[i])
    ases = np.array(ases)
    print(np.min(ases), np.max(ases), np.mean(ases), np.median(ases))
    indices = ases > 5
    '''
    for s in range(indices.shape[0]):
        for p in range(indices.shape[1]):
            if indices[s, p] and params[p] in samples[s]['OriParams'].keys() and samples[s]['OriParams'][params[p]].strip():
                print(f"##{params[p]}## ##{samples[s]['OriParams'][params[p]]}##")
