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

    def get_paths(self, white_samples):
        paths = []
        for sample in white_samples:
            path = sample['Path']
            paths.append(path)
        self.paths = list(set(paths))

    def value_lengths(self, sample):
        lengths = []
        for k in ['Path', 'NormParams']:
            if k in sample.keys():
                lengths.append(len(sample[k]))
            else:
                lengths.append(0)
        return lengths

    def init_param(self, white_samples):
        path_param = {}
        all_param = []

        for special in self.kvs:
            for s in white_samples:
                if special not in s.keys() or not s[special]:
                    continue
                params = s[special]
                for k in params.keys():
                    s_path = s['Path']
                    #path param
                    if s_path in path_param.keys():
                        if k not in path_param[s_path]:
                            path_param[s_path].append(k)
                    else:
                        path_param[s_path] = [k]

                    #all param
                    all_param.append(k)
        all_param = list(set(all_param))
        return path_param, all_param


if __name__ == '__main__':
    #vector_type = 'statistics'
    vector_type = 'tf-idf'
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

        if vector_type == 'statistics':
            v = []
            #Method index
            v.append(0 if sample['Method'] == 'GET' else 1 if sample['Method'] == 'POST' else -1)

            #key lengths
            v.extend(gv.value_lengths(sample))

            #vectorize count of never seen param key
            never_seen_key_count = 0
            for special in gv.kvs:
                if special in sample.keys() and sample[special]:
                    for subk, subv in sample[special].items():
                        if path not in path_param.keys():
                            if subk in all_param:
                                never_seen_key_count += 2
                            else:
                                never_seen_key_count += 5
            v.append(never_seen_key_count)
            path_buckets[path].append(v)

        elif vector_type == 'tf-idf':
            sample_param_str = ''
            for special in gv.kvs:
                if special in sample.keys() and sample[special]:
                    for subk, subv in sample[special].items():
                        sample_param_str += f'{subk}={subv} '
            path_buckets[path].append(sample_param_str)

    for path in path_buckets.keys():
        if vector_type == 'statistics':
            os.mkdir(f"path-{time.strftime('%Y-%m-%d')}")
            np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_x.npy", np.array(path_buckets[path]))
            np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_y.npy", np.array(path_ys[path]))
        elif vector_type == 'tf-idf':
            for path, strs in path_buckets.items():
                if not strs:
                    continue
                vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b\S\S+\b")
                try:
                    tfidf = vectorizer.fit_transform(strs)
                    print(path, len(vectorizer.vocabulary_))

                    if not os.path.exists(f"path-{time.strftime('%Y-%m-%d')}"):
                        os.mkdir(f"path-{time.strftime('%Y-%m-%d')}")
                    #save tfidf vectorizer for anomalious param extraction
                    joblib.dump(tfidf, f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_tfidf.m")
                    np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_x.npy", tfidf.toarray())
                    np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_y.npy", np.array(path_ys[path]))
                    np.save(f"path-{time.strftime('%Y-%m-%d')}/{path.replace('/', '~')}_index.npy", np.array(path_sample_indices[path]))
                except ValueError as ve:
                    print(path, ve)
