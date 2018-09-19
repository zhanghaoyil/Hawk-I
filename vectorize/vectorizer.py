"""
Copyright (c) Zacker
Hawk I
"""

from data.parse import ParseData
import numpy as np
import time, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

class Vectorizer:
    def __init__(self):
        self.kvs = ['NormParams']
        self.strs = ['Path']
        self.paths = []

    def get_paths(self, samples):
        '''
        Discover all possible valid paths
        '''
        paths = []
        for sample in samples:
            path = sample['Path']
            paths.append(path)
        self.paths = list(set(paths))

    def init_param(self, samples):
        '''
        Discover param keys in each path.
        '''
        path_param = {}
        all_param = []

        for special in self.kvs:
            for sample in samples:
                if special not in sample.keys() or not sample[special]:
                    continue
                params = sample[special]
                for k in params.keys():
                    path = sample['Path']
                    #path param
                    if path in path_param.keys():
                        if k not in path_param[path]:
                            path_param[path].append(k)
                    else:
                        path_param[path] = [k]

                    #all param
                    all_param.append(k)
        all_param = list(set(all_param))
        return path_param, all_param


if __name__ == '__main__':
    pd = ParseData('n1')
    white_samples = pd.samples
    pd = ParseData('a')
    black_samples = pd.samples
    #concat samples
    targets = [0] * len(white_samples) + [1]*len(black_samples)
    samples = white_samples + black_samples
    #init vectorizer
    gv = Vectorizer()
    gv.get_paths(white_samples)
    print(f'#Path count: {len(gv.paths)}')
    print('#Generating path-param database')
    path_param, all_param = gv.init_param(white_samples)
    print(f'#Found params count: {len(all_param)}')

    print('#Vectorizing Samples')
    path_buckets = {}
    path_ys = {}
    path_sample_indices = {}
    for sample in samples:
        #path
        path = sample['Path']
        if path not in gv.paths or 'jsp' not in path:
            continue
        if path not in path_buckets.keys():
            path_buckets[path] = []
        if path not in path_ys.keys():
            path_ys[path] = []
        if path not in path_sample_indices.keys():
            path_sample_indices[path] = []

        sample_index = samples.index(sample)
        target = targets[sample_index]
        path_ys[path].append(target)
        path_sample_indices[path].append(sample_index)

        sample_param_str = ''
        for special in gv.kvs:
            if special in sample.keys() and sample[special]:
                for subk, subv in sample[special].items():
                    sample_param_str += f'{subk}={subv} '
        path_buckets[path].append(sample_param_str)

    for path in path_buckets.keys():
        for path, strs in path_buckets.items():
            if not strs:
                continue
            vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b\S\S+\b")
            try:
                tfidf = vectorizer.fit_transform(strs)
                #putting same key's indices together
                param_index = {}
                for kv, index in vectorizer.vocabulary_.items():
                    k = kv.split('=')[0]
                    if k in param_index.keys():
                        param_index[k].append(index)
                    else:
                        param_index[k] = [index]
                #shrinking tfidf vectors
                vectors = []
                for vector in tfidf.toarray():
                    v = []
                    for param, index in param_index.items():
                        v.append(np.sum(vector[index]))
                    vectors.append(v)
                if not os.path.exists(f"path-{time.strftime('%Y-%m-%d')}"):
                    os.mkdir(f"path-{time.strftime('%Y-%m-%d')}")
                #save param index for anomalious param extraction
                np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_params.npy", np.array(list(param_index.keys())))
                np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_x.npy", np.array(vectors))
                np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_y.npy", np.array(path_ys[path]))
                np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_index.npy", np.array(path_sample_indices[path]))
            except ValueError as ve:
                print(path, ve)